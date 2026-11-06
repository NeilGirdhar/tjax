"""Tests for PCGrad."""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax import tree
from jax.flatten_util import ravel_pytree

from tjax import JaxArray, KeyArray
from tjax.gradient import SGD, Adam, PCGrad

Params = dict[str, JaxArray]
LOSS_IDS = ("first", "second", "third")
LR = 1.0


def _params() -> Params:
    return {"w": jnp.zeros((4, 3)), "b": jnp.zeros(3)}


def _gradient(seed: int = 1, loss_ids: tuple[str, ...] = LOSS_IDS) -> dict[str, Params]:
    keys = jr.split(jr.key(seed), len(loss_ids))
    return {
        loss_id: tree.map(lambda x, k=key: jr.normal(k, x.shape), _params())
        for loss_id, key in zip(loss_ids, keys, strict=True)
    }


def _combined(gradient: dict[str, Params], key: KeyArray) -> JaxArray:
    """Return the gradient that PCGrad passes to plain gradient descent, flattened."""
    params = tree.map(jnp.zeros_like, next(iter(gradient.values())))
    tx = PCGrad[Params](SGD(LR), key)
    update = jax.jit(lambda tx, state: tx.update(gradient, state, params)[0])
    flat, _ = ravel_pytree(update(tx, tx.init(params, tuple(gradient))))
    return -flat / LR


def _flat(gradient: dict[str, Params]) -> list[np.ndarray]:
    return [np.asarray(ravel_pytree(g)[0]) for g in gradient.values()]


def _reference(gradient: dict[str, Params], key: KeyArray) -> np.ndarray:
    """Follow Algorithm 1 of Yu et al. with the same projection orders."""
    g = _flat(gradient)
    n = len(g)
    _, subkey = jr.split(key)
    orders = [np.asarray(jr.permutation(k, n)) for k in jr.split(subkey, n)]
    total = np.zeros_like(g[0])
    for i in range(n):
        g_pc = g[i].copy()
        for j in orders[i]:
            if j != i and g_pc @ g[j] < 0:
                g_pc -= (g_pc @ g[j]) / (g[j] @ g[j]) * g[j]
        total += g_pc
    return total


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_matches_algorithm(seed: int) -> None:
    gradient = _gradient(seed)
    key = jr.key(seed)
    chex.assert_trees_all_close(_combined(gradient, key), _reference(gradient, key), rtol=1e-5)


def test_two_conflicting_gradients() -> None:
    first = jnp.array([1.0, 0.0])
    second = jnp.array([-1.0, 1.0])
    gradient = {
        "first": {"w": first, "b": jnp.zeros(0)},
        "second": {"w": second, "b": jnp.zeros(0)},
    }
    params = {"w": jnp.zeros(2), "b": jnp.zeros(0)}
    tx = PCGrad[Params](SGD(LR), jr.key(0))
    updates, _ = tx.update(gradient, tx.init(params, ("first", "second")), params)
    # first - (first . second) / |second|^2 second = (0.5, 0.5), and second loses its first
    # component to become (0, 1).
    chex.assert_trees_all_close(-updates["w"], jnp.array([0.5, 1.5]))


def test_gradients_that_agree_are_summed() -> None:
    base = tree.map(jnp.ones_like, _params())
    gradient = {"first": base, "second": tree.map(lambda x: 2.0 * x, base)}
    flat = _combined(gradient, jr.key(0))
    chex.assert_trees_all_close(flat, jnp.full(15, 3.0))


@pytest.mark.parametrize("seed", range(8))
def test_a_gradient_is_not_projected_onto_itself(seed: int) -> None:
    # Projecting the first gradient onto the others turns it past 90 degrees from where it started.
    gradient = {
        "first": {"w": jnp.array([1.0, 0.0])},
        "second": {"w": jnp.array([-1.0, 0.2])},
        "third": {"w": jnp.array([0.3, -1.0])},
    }
    key = jr.key(seed)
    chex.assert_trees_all_close(_combined(gradient, key), _reference(gradient, key), rtol=1e-5)


def test_key_advances() -> None:
    params = _params()
    tx = PCGrad[Params](SGD(LR), jr.key(0))
    state = tx.init(params, LOSS_IDS)
    _, new_state = tx.update(_gradient(), state, params)
    assert not jnp.array_equal(jr.key_data(state.statistics), jr.key_data(new_state.statistics))


def test_order_can_change_the_result() -> None:
    # The first gradient conflicts with both of the others, so the order of its projections matters.
    gradient = {
        "first": {"w": jnp.array([1.0, 0.0])},
        "second": {"w": jnp.array([-1.0, 0.2])},
        "third": {"w": jnp.array([-1.0, -2.0])},
    }
    results = {tuple(np.asarray(_combined(gradient, jr.key(i))).tolist()) for i in range(12)}
    assert len(results) > 1


def test_single_loss_is_the_inner_transformation() -> None:
    gradient = _gradient()["first"]
    params = _params()
    inner = Adam[Params](1e-2)
    expected, _ = inner.update(gradient, inner.init(params), params)
    tx = PCGrad[Params](inner, jr.key(0))
    updates, _ = tx.update({"first": gradient}, tx.init(params, ("first",)), params)
    chex.assert_trees_all_close(updates, expected)


def test_zero_gradient_stays_finite() -> None:
    gradient = _gradient()
    gradient["third"] = tree.map(jnp.zeros_like, gradient["third"])
    chex.assert_tree_all_finite(_combined(gradient, jr.key(0)))
