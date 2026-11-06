"""Tests for AveragedGradients."""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax import tree

from tjax import JaxArray
from tjax.gradient import (
    IMTLG,
    SGD,
    Adam,
    AveragedGradients,
    GradVac,
    MTAdam,
    PCGrad,
)

Params = dict[str, JaxArray]
LOSS_IDS = ("a", "b")
LR = 1.0
A = jnp.array([1.0, 0.5, 0.0, 0.0])
B = jnp.array([0.0, 0.5, 1.0, -0.5])
ZERO = jnp.zeros(4)
PARAMS: Params = {"w": jnp.zeros(4)}


def _alternating(steps: int) -> list[dict[str, Params]]:
    """Return gradients where the two losses never arrive at the same step."""
    return [
        {"a": {"w": A if i % 2 == 0 else ZERO}, "b": {"w": ZERO if i % 2 == 0 else B}}
        for i in range(steps)
    ]


def _reference(gradients: list[dict[str, Params]], decay: float) -> list[dict[str, np.ndarray]]:
    averages = dict.fromkeys(LOSS_IDS, np.zeros(4))
    result = []
    for gradient in gradients:
        averages = {
            i: decay * averages[i] + (1 - decay) * np.asarray(gradient[i]["w"]) for i in LOSS_IDS
        }
        result.append(dict(averages))
    return result


@pytest.mark.parametrize("decay", [0.0, 0.5, 0.9])
def test_inner_transformation_receives_the_averages(decay: float) -> None:
    gradients = _alternating(6)
    tx = AveragedGradients[Params](PCGrad(SGD(LR), jr.key(0)), decay=decay)
    state = tx.init(PARAMS, LOSS_IDS)
    for gradient, expected in zip(gradients, _reference(gradients, decay), strict=True):
        _, state = tx.update(gradient, state, PARAMS)
        for loss_id in LOSS_IDS:
            chex.assert_trees_all_close(state.averages[loss_id]["w"], expected[loss_id])


def test_zero_decay_is_the_inner_transformation() -> None:
    gradient = {"a": {"w": A}, "b": {"w": B}}
    inner = PCGrad[Params](SGD(LR), jr.key(0))
    expected, _ = inner.update(gradient, inner.init(PARAMS, LOSS_IDS), PARAMS)
    tx = AveragedGradients[Params](inner, decay=0.0)
    updates, _ = tx.update(gradient, tx.init(PARAMS, LOSS_IDS), PARAMS)
    chex.assert_trees_all_close(updates, expected)


def test_average_is_zero_until_a_loss_arrives() -> None:
    tx = AveragedGradients[Params](PCGrad(SGD(LR), jr.key(0)))
    _, state = tx.update(_alternating(1)[0], tx.init(PARAMS, LOSS_IDS), PARAMS)
    chex.assert_trees_all_close(state.averages["b"]["w"], ZERO)
    assert jnp.any(state.averages["a"]["w"] != 0.0)


def test_imtlg_updates_once_every_loss_has_arrived() -> None:
    tx = AveragedGradients[Params](IMTLG(SGD(LR)))
    state = tx.init(PARAMS, LOSS_IDS)
    first, *rest = _alternating(4)
    updates, state = tx.update(first, state, PARAMS)
    chex.assert_trees_all_close(updates["w"], ZERO, atol=1e-9)
    for gradient in rest:
        updates, state = tx.update(gradient, state, PARAMS)
    assert jnp.linalg.vector_norm(updates["w"]) > 0.0


def test_gradvac_learns_similarity_from_losses_that_never_coincide() -> None:
    cosine = jnp.vecdot(A, B) / (jnp.linalg.vector_norm(A) * jnp.linalg.vector_norm(B))
    tx = AveragedGradients[Params](GradVac(SGD(LR), jr.key(0), beta=0.1), decay=0.9)
    state = tx.init(PARAMS, LOSS_IDS)
    update = jax.jit(lambda tx, gradient, state: tx.update(gradient, state, PARAMS))
    for gradient in _alternating(400):
        _, state = update(tx, gradient, state)
    similarity = state.inner_state.statistics.similarities["w"][0, 1]
    chex.assert_trees_all_close(similarity, cosine, rtol=0.1)


def test_works_with_mtadam() -> None:
    tx = AveragedGradients[Params](MTAdam(1e-2))
    state = tx.init(PARAMS, LOSS_IDS)
    for gradient in _alternating(4):
        updates, state = tx.update(gradient, state, PARAMS)
    chex.assert_tree_all_finite(updates)


def test_decay_does_not_recompile() -> None:
    traces = 0
    gradient = _alternating(1)[0]

    @jax.jit
    def step(tx: AveragedGradients[Params]) -> Params:
        nonlocal traces
        traces += 1
        return tx.update(gradient, tx.init(PARAMS, LOSS_IDS), PARAMS)[0]

    step(AveragedGradients[Params](PCGrad(Adam(1e-2), jr.key(0)), decay=0.9))
    step(AveragedGradients[Params](PCGrad(Adam(1e-2), jr.key(0)), decay=0.5))
    assert traces == 1


def test_update_rejects_mismatched_losses() -> None:
    tx = AveragedGradients[Params](PCGrad(SGD(LR), jr.key(0)))
    state = tx.init(PARAMS, LOSS_IDS)
    with pytest.raises(ValueError, match="Expected gradients"):
        tx.update({"a": {"w": A}}, state, PARAMS)


def test_tree_structure_is_preserved() -> None:
    params = {"w": jnp.zeros((2, 2)), "b": jnp.zeros(2)}
    gradient = tree.map(jnp.ones_like, params)
    tx = AveragedGradients[dict[str, JaxArray]](PCGrad(SGD(LR), jr.key(0)))
    updates, _ = tx.update({"a": gradient, "b": gradient}, tx.init(params, LOSS_IDS), params)
    assert tree.structure(updates) == tree.structure(params)
