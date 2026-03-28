from __future__ import annotations

from collections.abc import Callable
from typing import Any

import chex
import jax.numpy as jnp
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


@pytest.mark.parametrize(
    "constructor",
    [
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
        lambda: NoisySGD[Params](1e-2),
        lambda: Novograd[Params](1e-2),
        lambda: OptimisticGradientDescent[Params](1e-2, 1e-2, 1e-2),
        lambda: RAdam[Params](1e-2),
        lambda: RMSProp[Params](1e-2),
        lambda: SGD[Params](1e-2),
        lambda: SM3[Params](1e-2),
        lambda: Yogi[Params](1e-2),
        lambda: DPSGD[Params](1e-2, 1.0, 0.0, 0),
    ],
)
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


def test_lbfgs_init() -> None:
    state = LBFGS[Params]().init(_params())
    assert state is not None
