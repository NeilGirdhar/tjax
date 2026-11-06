"""Tests that each combinator matches the optax combinator it wraps."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest
from jax import tree
from jax.typing import ArrayLike

from tjax import JaxArray, PyTree
from tjax.gradient import (
    SGD,
    Adam,
    ApplyIfFinite,
    ConditionallyMask,
    ConditionallyTransform,
    Flatten,
    Freeze,
    GenericGradientState,
    GradientTransformation,
    Lookahead,
    Masked,
    MeasureWithEMA,
    Monitor,
    MultiSteps,
    MultiTransform,
    NamedChain,
    Partition,
    PolyakSGD,
    Scale,
    SelectiveTransform,
    Snapshot,
    Stateless,
    StatelessWithTreeMap,
    Trace,
)

Params = dict[str, JaxArray]
STEPS = 6


def _params() -> Params:
    key_w, key_b = jr.split(jr.key(0))
    return {"w": jr.normal(key_w, (2, 3)), "b": jr.normal(key_b, (3,))}


def _gradients(*, with_nan: bool = False) -> list[Params]:
    keys = jr.split(jr.key(1), STEPS)
    gradients = [tree.map(lambda x, k=k: jr.normal(k, x.shape), _params()) for k in keys]
    if with_nan:
        gradients[2] = {**gradients[2], "b": gradients[2]["b"].at[0].set(jnp.nan)}
    return gradients


def _run_tjax(
    tx: GradientTransformation[Any, Any],
    params: PyTree,
    gradients: list[PyTree],
    extra_args: Mapping[str, Any],
) -> tuple[list[PyTree], PyTree]:
    @jax.jit
    def step(
        tx: GradientTransformation[Any, Any],
        gradient: PyTree,
        state: PyTree,
        params: PyTree,
        extra_args: Mapping[str, Any],
    ) -> tuple[PyTree, PyTree]:
        return tx.update(gradient, state, params, **extra_args)

    state = tx.init(params)
    all_updates = []
    for gradient in gradients:
        updates, state = step(tx, gradient, state, params, extra_args)
        all_updates.append(updates)
    return all_updates, state


def _run_optax(
    tx: optax.GradientTransformation,
    params: PyTree,
    gradients: list[PyTree],
    extra_args: Mapping[str, Any],
) -> list[PyTree]:
    state = tx.init(params)
    all_updates = []
    for gradient in gradients:
        if isinstance(tx, optax.GradientTransformationExtraArgs):
            updates, state = tx.update(gradient, state, params, **extra_args)
        else:
            updates, state = tx.update(gradient, state, params)
        all_updates.append(updates)
    return all_updates


def _assert_match(
    tjax_tx: GradientTransformation[Any, Any],
    optax_tx: optax.GradientTransformation,
    *,
    params: PyTree = None,
    gradients: list[PyTree] | None = None,
    extra_args: Mapping[str, Any] | None = None,
) -> PyTree:
    params = _params() if params is None else params
    gradients = _gradients() if gradients is None else gradients
    extra_args = {} if extra_args is None else extra_args
    tjax_updates, state = _run_tjax(tjax_tx, params, gradients, extra_args)
    optax_updates = _run_optax(optax_tx, params, gradients, extra_args)
    chex.assert_trees_all_close(tjax_updates, optax_updates, rtol=1e-5, atol=1e-6)
    return state


MASK = {"w": True, "b": False}
LABELS = {"w": "adam", "b": "sgd"}


def _scale_by_params(updates: PyTree, params: PyTree) -> PyTree:
    assert params is not None
    return tree.map(lambda u, p: u + 0.1 * p, updates, params)


def _double_minus_params(update: ArrayLike, param: ArrayLike | None) -> JaxArray:
    assert param is not None
    return 2.0 * jnp.asarray(update) - jnp.asarray(param)


def _even_step(step: ArrayLike, **extra_args: object) -> JaxArray:
    return jnp.asarray(step) % 2 == 0


CASES: list[tuple[str, Callable[[], GradientTransformation[Any, Any]], Callable[[], Any]]] = [
    (
        "masked",
        lambda: Masked[Params](Scale(2.0), MASK),
        lambda: optax.masked(optax.scale(2.0), MASK),
    ),
    (
        "masked-callable",
        lambda: Masked[Params](Adam(1e-2), lambda p: tree.map(lambda x: x.ndim > 1, p)),
        lambda: optax.masked(optax.adam(1e-2), lambda p: tree.map(lambda x: x.ndim > 1, p)),
    ),
    (
        "partition",
        lambda: Partition[Params]({"adam": Adam(1e-2), "sgd": SGD(1e-1)}, LABELS),
        lambda: optax.partition({"adam": optax.adam(1e-2), "sgd": optax.sgd(1e-1)}, LABELS),
    ),
    (
        "multi-transform",
        lambda: MultiTransform[Params]({"adam": Adam(1e-2), "sgd": SGD(1e-1)}, LABELS),
        lambda: optax.multi_transform({"adam": optax.adam(1e-2), "sgd": optax.sgd(1e-1)}, LABELS),
    ),
    ("freeze", lambda: Freeze[Params](MASK), lambda: optax.freeze(MASK)),
    (
        "selective-transform",
        lambda: SelectiveTransform[Params](Adam(1e-2), freeze_mask=MASK),
        lambda: optax.selective_transform(optax.adam(1e-2), freeze_mask=MASK),
    ),
    (
        "named-chain",
        lambda: NamedChain[Params](("trace", "scale"), [Trace(0.9), Scale(-0.1)]),
        lambda: optax.named_chain(("trace", optax.trace(0.9)), ("scale", optax.scale(-0.1))),
    ),
    ("flatten", lambda: Flatten[Params](Adam(1e-2)), lambda: optax.flatten(optax.adam(1e-2))),
    (
        "stateless",
        lambda: Stateless[Params](_scale_by_params),
        lambda: optax.stateless(_scale_by_params),
    ),
    (
        "stateless-with-tree-map",
        lambda: StatelessWithTreeMap[Params](_double_minus_params),
        lambda: optax.stateless_with_tree_map(_double_minus_params),
    ),
    (
        "conditionally-mask",
        lambda: ConditionallyMask[Params](Adam(1e-2), _even_step),
        lambda: optax.conditionally_mask(optax.adam(1e-2), _even_step),
    ),
    (
        "conditionally-transform",
        lambda: ConditionallyTransform[Params](Adam(1e-2), _even_step),
        lambda: optax.conditionally_transform(optax.adam(1e-2), _even_step),
    ),
    (
        "multi-steps",
        lambda: MultiSteps[Params](Adam(1e-2), 3),
        lambda: optax.MultiSteps(optax.adam(1e-2), 3).gradient_transformation(),
    ),
    (
        "snapshot",
        lambda: Snapshot[Params]("norm", optax.tree.norm),
        lambda: optax.snapshot("norm", optax.tree.norm),
    ),
    (
        "monitor",
        lambda: Monitor[Params](
            {"ema": MeasureWithEMA(optax.tree.norm, 0.9)}, {"norm": optax.tree.norm}
        ),
        lambda: optax.monitor(
            {"ema": optax.measure_with_ema(optax.tree.norm, 0.9), "norm": optax.tree.norm}
        ),
    ),
    (
        "measure-with-ema",
        lambda: MeasureWithEMA[Params](optax.tree.norm, 0.9),
        lambda: optax.measure_with_ema(optax.tree.norm, 0.9),
    ),
]


@pytest.mark.parametrize(
    ("tjax_tx", "optax_tx"), [case[1:] for case in CASES], ids=[case[0] for case in CASES]
)
def test_matches_optax(
    tjax_tx: Callable[[], GradientTransformation[Any, Any]], optax_tx: Callable[[], Any]
) -> None:
    _assert_match(tjax_tx(), optax_tx())


def test_apply_if_finite() -> None:
    gradients = _gradients(with_nan=True)
    _assert_match(
        ApplyIfFinite[Params](Adam(1e-2), 2),
        optax.apply_if_finite(optax.adam(1e-2), 2),
        gradients=gradients,
    )


def test_lookahead() -> None:
    params = optax.LookaheadParams.init_synced(_params())
    _assert_match(
        Lookahead[Any](SGD(1e-1), sync_period=3, slow_step_size=0.5),
        optax.lookahead(optax.sgd(1e-1), sync_period=3, slow_step_size=0.5),
        params=params,
    )


def test_extra_args_reach_inner_transform() -> None:
    _assert_match(
        Masked[Params](PolyakSGD(), MASK),
        optax.masked(optax.polyak_sgd(), MASK),
        extra_args={"value": jnp.asarray(2.0)},
    )


def test_multi_steps_has_updated() -> None:
    every_k = 3
    params = _params()
    tx = MultiSteps[Params](Adam(1e-2), every_k)
    reference = optax.MultiSteps(optax.adam(1e-2), every_k)

    @jax.jit
    def step(
        tx: MultiSteps[Params], gradient: Params, state: GenericGradientState
    ) -> tuple[GenericGradientState, JaxArray]:
        _, state = tx.update(gradient, state, params)
        return state, tx.has_updated(state)

    state = tx.init(params)
    reference_state = reference.init(params)
    assert not tx.has_updated(state)
    for i, gradient in enumerate(_gradients(), start=1):
        state, has_updated = step(tx, gradient, state)
        _, reference_state = reference.update(gradient, reference_state, params)
        assert bool(has_updated) == (i % every_k == 0)
        assert bool(has_updated) == bool(reference.has_updated(reference_state))


def test_named_chain_state_keys() -> None:
    tx = NamedChain[Params](("trace", "scale"), [Trace(0.9), Scale(-0.1)])
    assert list(tx.init(_params()).data) == ["trace", "scale"]
