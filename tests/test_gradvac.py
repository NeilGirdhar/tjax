"""Tests for GradVac."""

from __future__ import annotations

import copy

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax import tree

from tjax import JaxArray, KeyArray
from tjax.gradient import SGD, Adam, GradVac, PCGrad

Params = dict[str, JaxArray]
LOSS_IDS = ("first", "second", "third")
STEPS = 5
LR = 1.0
BETA = 0.2


def _params() -> Params:
    return {"w": jnp.zeros((4, 3)), "b": jnp.zeros(3)}


def _gradient(seed: int = 1, loss_ids: tuple[str, ...] = LOSS_IDS) -> dict[str, Params]:
    keys = jr.split(jr.key(seed), len(loss_ids))
    return {
        loss_id: tree.map(lambda x, k=key: jr.normal(k, x.shape), _params())
        for loss_id, key in zip(loss_ids, keys, strict=True)
    }


def _run(tx: GradVac[Params], gradients: list[dict[str, Params]]) -> list[Params]:
    params = tree.map(jnp.zeros_like, next(iter(gradients[0].values())))
    update = jax.jit(lambda tx, gradient, state: tx.update(gradient, state, params))
    state = tx.init(params)
    all_updates = []
    for gradient in gradients:
        updates, state = update(tx, gradient, state)
        all_updates.append(tree.map(lambda x: -x / LR, updates))
    return all_updates


def _adjust_block(
    g: np.ndarray, similarities: np.ndarray, key: KeyArray, beta: float
) -> tuple[np.ndarray, np.ndarray]:
    """Follow lines 9 to 17 of Algorithm 1 of Wang et al. for one parameter block."""
    n = len(g)
    similarities = similarities.copy()
    total = np.zeros_like(g[0])
    for i, order_key in enumerate(jr.split(key, n)):
        adjusted = g[i].copy()
        for j in np.asarray(jr.permutation(order_key, n)):
            if j == i:
                continue
            norm_i, norm_j = np.linalg.norm(adjusted), np.linalg.norm(g[j])
            phi = adjusted @ g[j] / (norm_i * norm_j)
            target = similarities[i, j]
            if phi < target:
                numerator = target * np.sqrt(1 - phi**2) - phi * np.sqrt(1 - target**2)
                adjusted += norm_i * numerator / (norm_j * np.sqrt(1 - target**2)) * g[j]
            similarities[i, j] = (1 - beta) * target + beta * phi
        total += adjusted
    return total, similarities


def _reference(
    gradients: list[dict[str, Params]], key: KeyArray, beta: float
) -> list[dict[str, np.ndarray]]:
    names = sorted(next(iter(gradients[0].values())))
    n = len(gradients[0])
    similarities = {name: np.zeros((n, n)) for name in names}
    result = []
    for gradient in gradients:
        key, subkey = jr.split(key)
        step = {}
        for name, block_key in zip(names, jr.split(subkey, len(names)), strict=True):
            shape = next(iter(gradient.values()))[name].shape
            g = np.stack([np.ravel(np.asarray(gradient[i][name])) for i in gradient])
            total, similarities[name] = _adjust_block(g, similarities[name], block_key, beta)
            step[name] = total.reshape(shape)
        result.append(step)
    return result


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_matches_algorithm_over_several_steps(seed: int) -> None:
    gradients = [_gradient(seed * 100 + step) for step in range(STEPS)]
    key = jr.key(seed)
    chex.assert_trees_all_close(
        _run(GradVac[Params](SGD(LR), key, beta=BETA, loss_ids=LOSS_IDS), gradients),
        _reference(gradients, key, BETA),
        rtol=1e-5,
        atol=1e-8,
    )


def test_first_step_is_pcgrad_for_two_losses() -> None:
    # The targets start at zero, which is PCGrad's special case.  Two losses leave no order to vary.
    gradient = {"first": {"w": jnp.array([1.0, 0.0])}, "second": {"w": jnp.array([-1.0, 1.0])}}
    params = {"w": jnp.zeros(2)}
    pcgrad = PCGrad[dict[str, JaxArray]](SGD(LR), jr.key(0), loss_ids=tuple(gradient))
    expected, _ = pcgrad.update(gradient, pcgrad.init(params), params)
    tx = GradVac[Params](SGD(LR), jr.key(0), loss_ids=tuple(gradient))
    updates, _ = tx.update(gradient, tx.init(params), params)
    chex.assert_trees_all_close(updates, expected)


@pytest.mark.parametrize("target", [-0.3, 0.2, 0.5, 0.9])
def test_adjustment_reaches_the_target_similarity(target: float) -> None:
    first = jnp.array([1.0, 0.0, 0.5])
    second = jnp.array([-1.0, 1.0, 0.25])
    gradient = {"first": {"w": first}, "second": {"w": second}}
    params = {"w": jnp.zeros(3)}
    tx = GradVac[Params](SGD(LR), jr.key(0), loss_ids=tuple(gradient))
    state = tx.init(params)
    # Only the first gradient has a target that it falls short of.
    similarities = {"w": jnp.array([[0.0, target], [-1.0, 0.0]])}
    statistics = copy.replace(state.statistics, similarities=similarities)
    state = copy.replace(state, statistics=statistics)
    updates, _ = tx.update(gradient, state, params)
    adjusted = -updates["w"] / LR - second
    cosine = jnp.vecdot(adjusted, second) / (
        jnp.linalg.vector_norm(adjusted) * jnp.linalg.vector_norm(second)
    )
    chex.assert_trees_all_close(cosine, target, rtol=1e-5)


def test_similar_gradients_are_left_alone() -> None:
    base = tree.map(jnp.ones_like, _params())
    gradient = {"first": base, "second": tree.map(lambda x: 2.0 * x, base)}
    (updates,) = _run(GradVac[Params](SGD(LR), jr.key(0), loss_ids=tuple(gradient)), [gradient])
    chex.assert_trees_all_close(updates, tree.map(lambda x: 3.0 * x, base))


def test_similarities_track_each_block_and_pair_separately() -> None:
    gradient = {
        "first": {"a": jnp.array([1.0, 0.0]), "b": jnp.array([1.0, 0.0])},
        "second": {"a": jnp.array([1.0, 0.0]), "b": jnp.array([0.0, 1.0])},
    }
    params = {"a": jnp.zeros(2), "b": jnp.zeros(2)}
    tx = GradVac[Params](SGD(LR), jr.key(0), beta=0.5, loss_ids=tuple(gradient))
    _, state = tx.update(gradient, tx.init(params), params)
    chex.assert_trees_all_close(
        state.statistics.similarities["a"], jnp.array([[0.0, 0.5], [0.5, 0.0]])
    )
    chex.assert_trees_all_close(state.statistics.similarities["b"], jnp.zeros((2, 2)))


def test_key_advances() -> None:
    params = _params()
    tx = GradVac[Params](SGD(LR), jr.key(0), loss_ids=LOSS_IDS)
    state = tx.init(params)
    _, new_state = tx.update(_gradient(), state, params)
    assert not jnp.array_equal(
        jr.key_data(state.statistics.key), jr.key_data(new_state.statistics.key)
    )


def test_single_loss_is_the_inner_transformation() -> None:
    gradient = _gradient()["first"]
    params = _params()
    inner = Adam[Params](1e-2)
    expected, _ = inner.update(gradient, inner.init(params), params)
    tx = GradVac[Params](inner, jr.key(0), loss_ids=("first",))
    updates, _ = tx.update({"first": gradient}, tx.init(params), params)
    chex.assert_trees_all_close(updates, expected)


def test_zero_gradient_stays_finite() -> None:
    gradients = [_gradient(step) for step in range(3)]
    for gradient in gradients:
        gradient["third"] = tree.map(jnp.zeros_like, gradient["third"])
    chex.assert_tree_all_finite(
        _run(GradVac[Params](SGD(LR), jr.key(0), loss_ids=LOSS_IDS), gradients)
    )


def test_extreme_targets_stay_finite() -> None:
    gradient = _gradient()
    params = _params()
    tx = GradVac[Params](SGD(LR), jr.key(0), loss_ids=LOSS_IDS)
    state = tx.init(params)
    similarities = tree.map(jnp.ones_like, state.statistics.similarities)
    statistics = copy.replace(state.statistics, similarities=similarities)
    updates, _ = tx.update(gradient, copy.replace(state, statistics=statistics), params)
    chex.assert_tree_all_finite(updates)


def test_beta_does_not_recompile() -> None:
    traces = 0
    gradient = _gradient()
    params = _params()

    @jax.jit
    def step(tx: GradVac[Params]) -> Params:
        nonlocal traces
        traces += 1
        return tx.update(gradient, tx.init(params), params)[0]

    step(GradVac[Params](SGD(LR), jr.key(0), beta=0.1, loss_ids=LOSS_IDS))
    step(GradVac[Params](SGD(LR), jr.key(1), beta=0.2, loss_ids=LOSS_IDS))
    assert traces == 1


def test_statistics_follow_the_parameter_dtype() -> None:
    params = {"w": jnp.zeros(3, jnp.float32)}
    gradient = {i: {"w": jnp.ones(3, jnp.float32)} for i in ("first", "second")}
    tx = GradVac[Params](SGD(LR), jr.key(0), loss_ids=tuple(gradient))
    state = tx.init(params)
    updates, new_state = tx.update(gradient, state, params)
    assert updates["w"].dtype == jnp.float32
    assert new_state.statistics.similarities["w"].dtype == jnp.float32
