from __future__ import annotations

from collections.abc import Callable
from typing import Any

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from jax import tree

from tjax.gradient import (
    DPSGD,
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
    Fromage,
    GradientTransformation,
    Lamb,
    Lion,
    NoisySGD,
    Novograd,
    OptimisticGradientDescent,
    RAdam,
    RMSProp,
    Schedule,
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
    lambda: AdamW[Params](1e-2),
    lambda: Adamax[Params](1e-2),
    lambda: AdamaxW[Params](1e-2),
    lambda: Fromage[Params](1e-2),
    lambda: Lamb[Params](1e-2),
    lambda: LARS[Params](1e-2),
    lambda: Lion[Params](1e-2),
    lambda: NoisySGD[Params](1e-2, key=jr.key(123)),
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
