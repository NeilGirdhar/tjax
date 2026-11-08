"""Tests for IMTL-G."""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax import tree
from jax.flatten_util import ravel_pytree

from tjax import JaxArray
from tjax.gradient import IMTLG, SGD, Adam

Params = dict[str, JaxArray]
LOSS_IDS = ("first", "second", "third")
SCALES = {"first": 1.0, "second": 50.0, "third": 0.02}
LR = 1.0


def _params() -> Params:
    key_w, key_b = jr.split(jr.key(0))
    return {"w": jr.normal(key_w, (4, 3)), "b": jr.normal(key_b, (3,))}


def _gradient(seed: int = 1) -> dict[str, Params]:
    keys = jr.split(jr.key(seed), len(LOSS_IDS))
    return {
        loss_id: tree.map(lambda x, k=key, s=SCALES[loss_id]: s * jr.normal(k, x.shape), _params())
        for loss_id, key in zip(LOSS_IDS, keys, strict=True)
    }


def _combined(gradient: dict[str, Params], loss_ids: tuple[str, ...] = LOSS_IDS) -> JaxArray:
    """Return the gradient that IMTL-G passes to plain gradient descent, flattened."""
    params = _params()
    tx = IMTLG[Params](SGD(LR), loss_ids=loss_ids)
    update = jax.jit(lambda tx, state: tx.update(gradient, state, params)[0])
    flat, _ = ravel_pytree(update(tx, tx.init(params)))
    return -flat / LR


def _flat(gradient: dict[str, Params]) -> np.ndarray:
    return np.stack([np.asarray(ravel_pytree(gradient[i])[0]) for i in LOSS_IDS])


def _reference(gradient: dict[str, Params]) -> np.ndarray:
    """Follow the closed form as written in LibMTL, with an explicit inverse."""
    g = _flat(gradient)
    u = g / np.linalg.norm(g, axis=1, keepdims=True)
    d_matrix = g[0:1] - g[1:]
    u_matrix = u[0:1] - u[1:]
    tail = g[0] @ u_matrix.T @ np.linalg.inv(d_matrix @ u_matrix.T)
    weights = np.concatenate([[1 - tail.sum()], tail])
    return weights @ g


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_matches_closed_form(seed: int) -> None:
    gradient = _gradient(seed)
    chex.assert_trees_all_close(_combined(gradient), _reference(gradient), rtol=1e-5)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_projections_onto_each_direction_are_equal(seed: int) -> None:
    gradient = _gradient(seed)
    g = _flat(gradient)
    directions = g / np.linalg.norm(g, axis=1, keepdims=True)
    projections = directions @ np.asarray(_combined(gradient))
    chex.assert_trees_all_close(projections, np.full(3, projections[0]), rtol=1e-5)


def test_single_loss_is_the_inner_transformation() -> None:
    gradient = _gradient()
    params = _params()
    inner = Adam[Params](1e-2)
    expected, _ = inner.update(gradient["first"], inner.init(params), params)
    tx = IMTLG[Params](inner, loss_ids=("first",))
    updates, _ = tx.update({"first": gradient["first"]}, tx.init(params), params)
    chex.assert_trees_all_close(updates, expected)


def test_identical_gradients_stay_finite() -> None:
    gradient = _gradient()
    gradient["second"] = gradient["first"]
    chex.assert_tree_all_finite(_combined(gradient))


def test_zero_gradient_stays_finite() -> None:
    gradient = _gradient()
    gradient["third"] = tree.map(jnp.zeros_like, gradient["third"])
    chex.assert_tree_all_finite(_combined(gradient))
