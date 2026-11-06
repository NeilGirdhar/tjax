"""Tests for gradient-statistics annealing."""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax import tree

from tjax import JaxArray
from tjax.gradient import SGD, Adam, GradientStatisticsAnnealing

Params = dict[str, JaxArray]
LOSS_IDS = ("residual", "boundary", "initial")
SCALES = {"residual": 100.0, "boundary": 1.0, "initial": 0.01}
STEPS = 6
LR = 1e-2


def _params() -> Params:
    key_w, key_b = jr.split(jr.key(0))
    return {"w": jr.normal(key_w, (4, 3)), "b": jr.normal(key_b, (3,))}


def _gradients() -> list[dict[str, Params]]:
    def one_step(step: int) -> dict[str, Params]:
        keys = jr.split(jr.key(step + 1), len(LOSS_IDS))
        return {
            loss_id: tree.map(
                lambda x, k=key, s=SCALES[loss_id]: s * jr.normal(k, x.shape), _params()
            )
            for loss_id, key in zip(LOSS_IDS, keys, strict=True)
        }

    return [one_step(step) for step in range(STEPS)]


def _run(
    tx: GradientStatisticsAnnealing[Params],
    gradients: list[dict[str, Params]],
    loss_ids: tuple[str, ...] = LOSS_IDS,
) -> list[Params]:
    params = _params()
    update = jax.jit(lambda tx, g, s: tx.update(g, s, params))
    state = tx.init(params, loss_ids)
    all_updates = []
    for gradient in gradients:
        updates, state = update(tx, gradient, state)
        all_updates.append(updates)
    return all_updates


def _flat(gradient: Params) -> np.ndarray:
    return np.concatenate([np.ravel(gradient[name]) for name in sorted(gradient)])


def _reference(gradients: list[dict[str, Params]], alpha: float) -> list[dict[str, np.ndarray]]:
    """Follow algorithm 1 of Wang et al. with plain gradient descent.

    As in the authors' code, the mean is of the gradient of the weighted loss.
    """
    weights = dict.fromkeys(LOSS_IDS[1:], 1.0)
    result = []
    for gradient in gradients:
        peak = np.max(np.abs(_flat(gradient[LOSS_IDS[0]])))
        for i in LOSS_IDS[1:]:
            estimate = peak / np.mean(np.abs(weights[i] * _flat(gradient[i])))
            weights[i] = (1 - alpha) * weights[i] + alpha * estimate
        result.append(
            {
                name: -LR
                * (
                    np.asarray(gradient[LOSS_IDS[0]][name])
                    + sum(weights[i] * np.asarray(gradient[i][name]) for i in LOSS_IDS[1:])
                )
                for name in gradient[LOSS_IDS[0]]
            }
        )
    return result


@pytest.mark.parametrize("alpha", [0.1, 0.9])
def test_matches_the_algorithm(alpha: float) -> None:
    gradients = _gradients()
    tx = GradientStatisticsAnnealing[Params](SGD(LR), alpha=alpha)
    chex.assert_trees_all_close(_run(tx, gradients), _reference(gradients, alpha), rtol=1e-5)


def test_single_loss_is_the_inner_transformation() -> None:
    gradients = [{"residual": g["residual"]} for g in _gradients()]
    inner = Adam[Params](LR)
    state = inner.init(_params())
    expected = []
    for gradient in gradients:
        updates, state = inner.update(gradient["residual"], state, _params())
        expected.append(updates)
    tx = GradientStatisticsAnnealing[Params](inner)
    chex.assert_trees_all_close(_run(tx, gradients, ("residual",)), expected)


def test_zero_gradient_keeps_its_weight() -> None:
    gradient = _gradients()[0]
    gradient["boundary"] = tree.map(jnp.zeros_like, gradient["boundary"])
    tx = GradientStatisticsAnnealing[Params](SGD(LR))
    params = _params()
    updates, state = tx.update(gradient, tx.init(params, LOSS_IDS), params)
    chex.assert_tree_all_finite(updates)
    assert float(state.statistics[0]) == 1.0


def test_hyperparameters_do_not_recompile() -> None:
    traces = 0
    gradient = _gradients()[0]
    params = _params()

    @jax.jit
    def step(tx: GradientStatisticsAnnealing[Params]) -> Params:
        nonlocal traces
        traces += 1
        return tx.update(gradient, tx.init(params, LOSS_IDS), params)[0]

    step(GradientStatisticsAnnealing[Params](SGD(1e-2), alpha=0.9))
    step(GradientStatisticsAnnealing[Params](SGD(1e-3), alpha=0.5))
    assert traces == 1


def test_update_rejects_mismatched_losses() -> None:
    tx = GradientStatisticsAnnealing[Params](SGD(LR))
    state = tx.init(_params(), LOSS_IDS)
    with pytest.raises(ValueError, match="Expected gradients"):
        tx.update({"residual": _params()}, state, _params())


def test_weights_follow_the_parameter_dtype() -> None:
    params = {"w": jnp.zeros(3, jnp.float32)}
    gradient = {i: {"w": jnp.ones(3, jnp.float32)} for i in ("residual", "boundary")}
    tx = GradientStatisticsAnnealing[Params](SGD(LR))
    state = tx.init(params, tuple(gradient))
    updates, new_state = tx.update(gradient, state, params)
    assert updates["w"].dtype == jnp.float32
    assert new_state.statistics.dtype == jnp.float32
