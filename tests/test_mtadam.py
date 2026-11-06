"""Tests for MTAdam and its base class."""

from __future__ import annotations

from collections.abc import Mapping

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pytest
from jax import tree

from tjax import JaxArray
from tjax.gradient import (
    Adam,
    MTAdam,
    MTAdamState,
    MultiLossGradientTransformation,
    Schedule,
)

Params = dict[str, JaxArray]
LOSS_IDS = ("anchor", "large", "small")
SCALES = {"anchor": 1.0, "large": 1000.0, "small": 1e-3}
STEPS = 6
LR = 1e-2
MIN_NORM = 1e-10


def _params() -> Params:
    key_w, key_b = jr.split(jr.key(0))
    return {"w": jr.normal(key_w, (4, 3)), "b": jr.normal(key_b, (3,))}


def _gradients(loss_ids: tuple[str, ...] = LOSS_IDS) -> list[dict[str, Params]]:
    def one_step(step: int) -> dict[str, Params]:
        keys = jr.split(jr.key(step + 1), len(loss_ids))
        return {
            loss_id: tree.map(
                lambda x, k=key, s=SCALES[loss_id]: s * jr.normal(k, x.shape), _params()
            )
            for loss_id, key in zip(loss_ids, keys, strict=True)
        }

    return [one_step(step) for step in range(STEPS)]


def _run(
    tx: MTAdam[Params], gradients: list[dict[str, Params]], loss_ids: tuple[str, ...] = LOSS_IDS
) -> list[Params]:
    @jax.jit
    def step(
        tx: MTAdam[Params], gradient: Mapping[str, Params], state: MTAdamState[Params]
    ) -> tuple[Params, MTAdamState[Params]]:
        return tx.update(gradient, state, _params())

    state = tx.init(_params(), loss_ids)
    all_updates = []
    for gradient in gradients:
        updates, state = step(tx, gradient, state)
        all_updates.append(updates)
    return all_updates


def _reference_parameter(
    gradients: list[dict[str, np.ndarray]], loss_ids: tuple[str, ...]
) -> list[np.ndarray]:
    """Run a NumPy port of the authors' ``update_weights`` on one parameter."""
    b1, b2, b3, eps = 0.9, 0.999, 0.9, 1e-8
    shape = gradients[0][loss_ids[0]].shape
    m = {i: np.zeros(shape) for i in loss_ids}
    v = {i: np.zeros(shape) for i in loss_ids}
    n = dict.fromkeys(loss_ids, 1.0)
    result = []
    for t, gradient in enumerate(gradients, start=1):
        denominators = []
        for i in loss_ids:
            g = gradient[i]
            norm = float(np.linalg.norm(g))
            n[i] = norm if t == 1 else n[i] * b3 + (1 - b3) * norm
            if n[i] > MIN_NORM:
                anchor = next(j for j in loss_ids if n[j] > MIN_NORM)
                g = n[anchor] * g / n[i]
            m[i] = b1 * m[i] + (1 - b1) * g
            v[i] = b2 * v[i] + (1 - b2) * g * g
            denominators.append(np.sqrt(v[i]) / np.sqrt(1 - b2**t) + eps)
        largest = np.maximum.reduce(denominators)
        result.append(sum(-LR / (1 - b1**t) * (m[i] / largest) for i in loss_ids))
    return result


def _reference(
    gradients: list[dict[str, Params]], loss_ids: tuple[str, ...]
) -> list[dict[str, np.ndarray]]:
    per_parameter = {
        name: _reference_parameter(
            [{i: np.asarray(g[i][name]) for i in loss_ids} for g in gradients], loss_ids
        )
        for name in _params()
    }
    return [{name: updates[t] for name, updates in per_parameter.items()} for t in range(STEPS)]


def test_matches_reference_implementation() -> None:
    gradients = _gradients()
    chex.assert_trees_all_close(
        _run(MTAdam[Params](LR), gradients), _reference(gradients, LOSS_IDS), rtol=1e-5
    )


def test_zero_gradients_match_reference_implementation() -> None:
    gradients = _gradients()
    for gradient in gradients[:3]:
        gradient["anchor"] = tree.map(jnp.zeros_like, gradient["anchor"])
    updates = _run(MTAdam[Params](LR), gradients)
    chex.assert_tree_all_finite(updates)
    chex.assert_trees_all_close(updates, _reference(gradients, LOSS_IDS), rtol=1e-5)


def test_single_loss_is_adam() -> None:
    gradients = _gradients(("anchor",))
    tx = Adam[Params](LR)
    state = tx.init(_params())
    expected = []
    for gradient in gradients:
        updates, state = tx.update(gradient["anchor"], state, _params())
        expected.append(updates)
    chex.assert_trees_all_close(
        _run(MTAdam[Params](LR), gradients, ("anchor",)), expected, rtol=1e-5
    )


def test_rescaling_a_loss_has_no_effect() -> None:
    gradients = _gradients()
    rescaled = [{**g, "large": tree.map(lambda x: 1234.5 * x, g["large"])} for g in gradients]
    tx = MTAdam[Params](LR)
    chex.assert_trees_all_close(_run(tx, gradients), _run(tx, rescaled), rtol=1e-5)


def test_mapping_order_has_no_effect() -> None:
    gradients = _gradients()
    reordered = [dict(reversed(g.items())) for g in gradients]
    tx = MTAdam[Params](LR)
    chex.assert_trees_all_close(_run(tx, gradients), _run(tx, reordered))


def test_anchor_is_the_first_loss_id() -> None:
    gradients = _gradients()
    tx = MTAdam[Params](LR)
    other_anchor = ("large", "anchor", "small")
    chex.assert_trees_all_close(
        _run(tx, gradients, other_anchor), _reference(gradients, other_anchor), rtol=1e-5
    )


def test_schedule() -> None:
    gradients = _gradients()
    scheduled = _run(MTAdam[Params](Schedule(lambda _: LR)), gradients)
    chex.assert_trees_all_close(scheduled, _run(MTAdam[Params](LR), gradients))


def test_hyperparameters_do_not_recompile() -> None:
    traces = 0
    gradient = _gradients()[0]

    @jax.jit
    def step(tx: MTAdam[Params]) -> Params:
        nonlocal traces
        traces += 1
        updates, _ = tx.update(gradient, tx.init(_params(), LOSS_IDS), _params())
        return updates

    step(MTAdam[Params](1e-2, b3=0.9))
    step(MTAdam[Params](1e-3, b3=0.8))
    assert traces == 1


def test_differentiates_with_respect_to_learning_rate() -> None:
    gradient = _gradients()[0]

    def total(learning_rate: JaxArray) -> JaxArray:
        tx = MTAdam[Params](learning_rate)
        updates, _ = tx.update(gradient, tx.init(_params(), LOSS_IDS), _params())
        return jnp.asarray(optax.tree.sum(updates))

    derivative = jax.grad(total)(jnp.asarray(LR))
    chex.assert_trees_all_close(derivative, total(jnp.asarray(LR)) / LR, rtol=1e-5)


def test_update_is_additive() -> None:
    gradients = _gradients()
    params = _params()
    tx = MTAdam[Params](LR)
    updates, _ = tx.update(gradients[0], tx.init(params, LOSS_IDS), params)
    chex.assert_trees_all_equal_shapes(optax.apply_updates(params, updates), params)


@pytest.mark.parametrize("loss_ids", [(), ("a", "a")], ids=["empty", "duplicate"])
def test_init_rejects_bad_loss_ids(loss_ids: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="nonempty and distinct"):
        MTAdam[Params](LR).init(_params(), loss_ids)


def test_update_rejects_mismatched_losses() -> None:
    tx = MTAdam[Params](LR)
    state = tx.init(_params(), LOSS_IDS)
    with pytest.raises(ValueError, match="Expected gradients"):
        tx.update({"anchor": _params()}, state, _params())


def test_base_class_is_abstract() -> None:
    with pytest.raises(TypeError, match="abstract"):
        MultiLossGradientTransformation()  # ty: ignore[call-non-callable]
