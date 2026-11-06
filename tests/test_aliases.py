from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast, override

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest
from jax import tree

from tjax.dataclasses import dataclass
from tjax.gradient import (
    DPSGD,
    LAMB,
    LARS,
    LBFGS,
    SGD,
    SM3,
    AdaBelief,
    AdaDelta,
    AdaFactor,
    AdaGrad,
    Adam,
    Adamax,
    AdamaxW,
    AdamW,
    ChainedGradientState,
    ChainedGradientTransformation,
    DiagHessianGradientTransformation,
    Fromage,
    FromOptax,
    GenericGradientState,
    GradientState,
    GradientTransformation,
    Lion,
    Masked,
    NoisySGD,
    Novograd,
    OptimisticGradientDescent,
    PolyakSGD,
    RAdam,
    RMSProp,
    Scale,
    ScaleByBacktrackingLinesearch,
    ScaleByZoomLinesearch,
    Schedule,
    SMDGradient,
    Yogi,
)

Params = dict[str, jnp.ndarray]


def _params() -> Params:
    return {"w": jnp.array([1.0, 2.0])}


def _gradient() -> Params:
    return {"w": jnp.array([0.1, 0.2])}


CONSTRUCTORS: list[Callable[[], Any]] = [
    lambda: AdaBelief[Params](1e-2),
    AdaDelta[Params],
    AdaFactor[Params],
    lambda: AdaGrad[Params](1e-2),
    lambda: Adam[Params](1e-2),
    lambda: Adam[Params](Schedule(lambda _: 1e-2)),
    lambda: Adam[Params](Schedule(optax.linear_schedule(1e-2, 0.0, 10))),
    lambda: AdamW[Params](1e-2),
    lambda: AdamW[Params](Schedule(lambda _: 1e-2), weight_decay=Schedule(lambda _: 1e-4)),
    lambda: Adamax[Params](1e-2),
    lambda: AdamaxW[Params](1e-2),
    lambda: Fromage[Params](1e-2),
    lambda: LAMB[Params](1e-2),
    lambda: LARS[Params](1e-2),
    lambda: Lion[Params](1e-2),
    lambda: NoisySGD[Params](1e-2, key=jr.key(123)),
    lambda: NoisySGD[Params](1e-2, key=123),
    lambda: Novograd[Params](1e-2),
    lambda: OptimisticGradientDescent[Params](1e-2, 1e-2, 1e-2),
    lambda: RAdam[Params](1e-2),
    lambda: RMSProp[Params](1e-2),
    lambda: SGD[Params](1e-2),
    lambda: SM3[Params](1e-2),
    lambda: Yogi[Params](1e-2),
    lambda: DPSGD[Params](1e-2, 1.0, 0.0, 0),
]


@pytest.mark.parametrize("constructor", CONSTRUCTORS)
def test_aliases_smoke(
    constructor: Callable[[], Any],
) -> None:
    params = _params()
    gradient = _gradient()
    tx = constructor()
    state = tx.init(params)
    updates, new_state = tx.update(gradient, state, params)

    chex.assert_tree_all_finite(updates)
    assert type(new_state) is type(state)
    assert tree.structure(updates) == tree.structure(gradient)


@pytest.mark.parametrize("constructor", CONSTRUCTORS)
def test_aliases_jit(
    constructor: Callable[[], Any],
) -> None:
    @jax.jit
    def step(tx: GradientTransformation[Any, Params], gradient: Params, params: Params) -> Params:
        state = tx.init(params)
        updates, _ = tx.update(gradient, state, params)
        return updates

    updates = step(constructor(), _gradient(), _params())
    chex.assert_tree_all_finite(updates)


def _mask(params: Params) -> dict[str, bool]:
    return {"w": True, "b": False}


@pytest.mark.parametrize("constructor", [AdamW[Params], AdamaxW[Params], Lion[Params]])
@pytest.mark.parametrize("mask", [_mask, {"w": True, "b": False}])
def test_mask_jit(constructor: Callable[..., Any], mask: object) -> None:
    @jax.jit
    def step(tx: GradientTransformation[Any, Params], gradient: Params, params: Params) -> Params:
        state = tx.init(params)
        updates, _ = tx.update(gradient, state, params)
        return updates

    params = {"w": jnp.array([1.0, 2.0]), "b": jnp.array([3.0])}
    gradient = {"w": jnp.array([0.1, 0.2]), "b": jnp.array([0.3])}
    masked = step(constructor(1e-2, weight_decay=1e2, mask=mask), gradient, params)
    unmasked = step(constructor(1e-2, weight_decay=1e2, mask=False), gradient, params)
    decayed = step(constructor(1e-2, weight_decay=1e2), gradient, params)
    chex.assert_trees_all_close(masked["w"], decayed["w"])
    chex.assert_trees_all_close(masked["b"], unmasked["b"])


def test_lbfgs_init() -> None:
    state = LBFGS[Params]().init(_params())
    assert state is not None


CONVERGED_LOSS = 1e-4


def _loss(params: Params) -> jax.Array:
    return jnp.sum(jnp.square(params["w"] - 3.0))


@pytest.mark.parametrize(
    "tx",
    [
        LBFGS[Params](),
        LBFGS[Params](learning_rate=0.5, linesearch=None),
        LBFGS[Params](linesearch=ScaleByBacktrackingLinesearch(15)),
        LBFGS[Params](linesearch=FromOptax(optax.scale_by_zoom_linesearch(15))),
        PolyakSGD[Params](),
    ],
    ids=[
        "lbfgs-linesearch",
        "lbfgs-no-linesearch",
        "lbfgs-tjax-linesearch",
        "lbfgs-from-optax",
        "polyak-sgd",
    ],
)
def test_value_transforms_jit(tx: LBFGS[Params] | PolyakSGD[Params]) -> None:
    @jax.jit
    def step(
        tx: LBFGS[Params] | PolyakSGD[Params], params: Params, state: GenericGradientState
    ) -> tuple[Params, GenericGradientState]:
        value, grad = jax.value_and_grad(_loss)(params)
        if isinstance(tx, LBFGS):
            updates, state = tx.update(grad, state, params, value=value, grad=grad, value_fn=_loss)
        else:
            updates, state = tx.update(grad, state, params, value=value)
        return tree.map(jnp.add, params, updates), state

    params = _params()
    state = tx.init(params)
    for _ in range(10):
        params, state = step(tx, params, state)
    assert _loss(params) < CONVERGED_LOSS


def test_polyak_sgd_requires_value() -> None:
    tx = PolyakSGD[Params]()
    with pytest.raises(TypeError, match="value"):
        tx.update(_gradient(), tx.init(_params()), _params())


@pytest.mark.parametrize(
    "transforms",
    [
        [LBFGS[Params]()],
        [PolyakSGD[Params](), Scale[Params](1.0)],
        [ChainedGradientTransformation[Params]([PolyakSGD[Params]()])],
    ],
    ids=["lbfgs", "polyak-sgd-then-scale", "nested"],
)
def test_chain_forwards_extra_args(transforms: list[GradientTransformation[Any, Params]]) -> None:
    @jax.jit
    def step(
        tx: ChainedGradientTransformation[Params], params: Params, state: ChainedGradientState
    ) -> tuple[Params, ChainedGradientState]:
        value, grad = jax.value_and_grad(_loss)(params)
        updates, state = tx.update(grad, state, params, value=value, grad=grad, value_fn=_loss)
        return tree.map(jnp.add, params, updates), state

    tx = ChainedGradientTransformation[Params](transforms)
    params = _params()
    state = tx.init(params)
    for _ in range(10):
        params, state = step(tx, params, state)
    assert _loss(params) < CONVERGED_LOSS


def test_chain_update_without_parameters() -> None:
    tx = ChainedGradientTransformation[Params]([Scale[Params](2.0)])
    updates, _ = tx.update(_gradient(), tx.init(_params()))
    chex.assert_trees_all_close(updates, tree.map(lambda x: 2.0 * x, _gradient()))


def test_linesearch_hyperparameters_do_not_recompile() -> None:
    traces = 0

    @jax.jit
    def step(tx: LBFGS[Params], params: Params) -> Params:
        nonlocal traces
        traces += 1
        value, grad = jax.value_and_grad(_loss)(params)
        updates, _ = tx.update(
            grad, tx.init(params), params, value=value, grad=grad, value_fn=_loss
        )
        return updates

    for increase_factor in (2.0, 3.0):
        linesearch = ScaleByZoomLinesearch[Params](15, increase_factor=increase_factor)
        step(LBFGS[Params](linesearch=linesearch), _params())
    assert traces == 1


def test_value_and_grad_from_state_finds_linesearch_state() -> None:
    tx = LBFGS[Params]()
    params = _params()
    value, grad = jax.value_and_grad(_loss)(params)
    _, state = tx.update(grad, tx.init(params), params, value=value, grad=grad, value_fn=_loss)
    cached_value, _ = optax.value_and_grad_from_state(_loss)(params, state=state)
    assert jnp.isfinite(cached_value)


@dataclass
class _RecordingState(GradientState):
    value: jax.Array


@dataclass
class _ValueScaledDescent(DiagHessianGradientTransformation[_RecordingState, Params]):
    """A transformation that uses an extra argument at the bottom of the class hierarchy."""

    @override
    def init(self, parameters: Params) -> _RecordingState:
        return _RecordingState(jnp.zeros(()))

    @override
    def diag_hessian_update(
        self,
        gradient: Params,
        state: _RecordingState,
        parameters: Params | None,
        hessian_vector_product: Callable[[Params], Params],
        hessian_diagonal: Params,
        **extra_args: object,
    ) -> tuple[Params, _RecordingState]:
        value = cast("jax.Array", extra_args["value"])
        return tree.map(lambda g: value * g, gradient), _RecordingState(value)


def _update_with_value(
    tx: GradientTransformation[Any, Params], state: GradientState, value: jax.Array
) -> Params:
    # This is typed against the base class, which must accept extra arguments.
    updates, _ = tx.update(_gradient(), state, _params(), value=value)
    return updates


@pytest.mark.parametrize(
    "tx",
    [
        Scale[Params](2.0),
        PolyakSGD[Params](),
        SMDGradient[Params](),
        ChainedGradientTransformation[Params]([SMDGradient[Params](), PolyakSGD[Params]()]),
        Masked[Params](SMDGradient[Params](), {"w": True}),
    ],
    ids=["ignored", "used", "hvp-ignored", "chain", "through-optax"],
)
def test_every_transformation_accepts_extra_args(tx: GradientTransformation[Any, Params]) -> None:
    updates = _update_with_value(tx, tx.init(_params()), jnp.asarray(2.0))
    chex.assert_tree_all_finite(updates)


def test_extra_args_reach_diag_hessian_update() -> None:
    tx = _ValueScaledDescent()
    updates = _update_with_value(tx, tx.init(_params()), jnp.asarray(2.0))
    chex.assert_trees_all_close(updates, tree.map(lambda g: 2.0 * g, _gradient()))
