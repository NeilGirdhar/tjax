"""Tests for CAGrad."""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax import tree
from jax.flatten_util import ravel_pytree
from scipy.optimize import minimize

from tjax import JaxArray
from tjax._src.gradient.cagrad import (  # ruff: ignore[import-private-name]
    conflict_averse_weights,
)
from tjax.gradient import SGD, CAGrad

Params = dict[str, JaxArray]
LOSS_IDS = ("first", "second", "third")
LR = 1.0
TOLERANCE = 1e-5


def _params() -> Params:
    return {"w": jnp.zeros((4, 3)), "b": jnp.zeros(3)}


def _gradient(seed: int = 1) -> dict[str, Params]:
    keys = jr.split(jr.key(seed), len(LOSS_IDS))
    return {
        loss_id: tree.map(lambda x, k=key: jr.normal(k, x.shape), _params())
        for loss_id, key in zip(LOSS_IDS, keys, strict=True)
    }


def _combined(gradient: dict[str, Params], tx: CAGrad[Params]) -> JaxArray:
    """Return the direction that CAGrad passes to plain gradient descent, flattened."""
    params = tree.map(jnp.zeros_like, next(iter(gradient.values())))
    update = jax.jit(lambda tx, state: tx.update(gradient, state, params)[0])
    flat, _ = ravel_pytree(update(tx, tx.init(params)))
    return -flat / LR


def _flat(gradient: dict[str, Params]) -> np.ndarray:
    return np.stack([np.asarray(ravel_pytree(g)[0]) for g in gradient.values()])


def _objective(w: np.ndarray, gram: np.ndarray, radius: float) -> float:
    b = np.ones(len(w)) / len(w)
    return float(w @ gram @ b + radius * np.sqrt(w @ gram @ w + 1e-8))


def _slsqp_weights(gram: np.ndarray, radius: float) -> np.ndarray:
    k = len(gram)
    start = np.ones(k) / k
    return minimize(
        _objective,
        start,
        args=(gram, radius),
        bounds=[(0, 1)] * k,
        constraints={"type": "eq", "fun": lambda x: 1 - x.sum()},
    ).x


def _reference(g: np.ndarray, c: float) -> np.ndarray:
    """Follow the authors' ``cagrad`` function, with SciPy's solver."""
    gram = g @ g.T
    radius = c * np.sqrt(gram.mean() + 1e-8) + 1e-8
    weighted = _slsqp_weights(gram, radius) @ g
    multiplier = radius / (np.linalg.norm(weighted) + 1e-8)
    return (g.mean(axis=0) + multiplier * weighted) / (1 + c**2)


@pytest.mark.parametrize("c", [0.2, 0.5, 0.9])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_matches_authors_code(seed: int, c: float) -> None:
    gradient = _gradient(seed)
    expected = _reference(_flat(gradient), c)
    actual = _combined(gradient, CAGrad[Params](SGD(LR), loss_ids=LOSS_IDS, c=c))
    chex.assert_trees_all_close(actual, expected, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("kind", ["random", "scaled", "conflicting", "nearly-parallel"])
@pytest.mark.parametrize("n_losses", [2, 3, 5, 8])
def test_solver_is_as_good_as_slsqp(kind: str, n_losses: int) -> None:
    rng = np.random.default_rng(n_losses)
    size = 50
    base = rng.normal(size=size)
    noise = rng.normal(size=(n_losses, size))
    g = {
        "random": noise,
        "scaled": noise * 10.0 ** rng.uniform(-3, 3, size=(n_losses, 1)),
        "conflicting": np.stack([(-1) ** i * base for i in range(n_losses)]) + 0.05 * noise,
        "nearly-parallel": base * rng.uniform(0.5, 2, size=(n_losses, 1)) + 1e-3 * noise,
    }[kind]
    gram = g @ g.T
    radius = 0.5 * np.sqrt(gram.mean() + 1e-8) + 1e-8
    weights = np.asarray(conflict_averse_weights(jnp.asarray(gram), jnp.asarray(radius)))
    chex.assert_trees_all_close(weights.sum(), 1.0, rtol=1e-9)
    assert weights.min() >= 0.0
    ours = _objective(weights, gram, radius)
    theirs = _objective(_slsqp_weights(gram, radius), gram, radius)
    assert ours <= theirs + TOLERANCE * max(abs(theirs), np.trace(gram) / n_losses * 1e-6)


def test_direction_stays_in_the_ball_around_the_average() -> None:
    gradient = _gradient(4)
    g = _flat(gradient)
    c = 0.5
    average = g.mean(axis=0)
    direction = np.asarray(
        _combined(gradient, CAGrad[Params](SGD(LR), loss_ids=LOSS_IDS, c=c, rescale="none"))
    )
    chex.assert_trees_all_close(
        np.linalg.norm(direction - average), c * np.linalg.norm(average), rtol=1e-4
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_improves_the_worst_loss(seed: int) -> None:
    gradient = _gradient(seed)
    g = _flat(gradient)
    direction = np.asarray(
        _combined(gradient, CAGrad[Params](SGD(LR), loss_ids=LOSS_IDS, rescale="none"))
    )
    assert (g @ direction).min() >= (g @ g.mean(axis=0)).min()


def test_zero_radius_is_the_average_gradient() -> None:
    gradient = _gradient()
    direction = _combined(gradient, CAGrad[Params](SGD(LR), loss_ids=LOSS_IDS, c=0.0))
    chex.assert_trees_all_close(direction, _flat(gradient).mean(axis=0), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(
    ("rescale", "divisor"), [("none", 1.0), ("linear", 1.5), ("quadratic", 1.25)]
)
def test_rescale(rescale: str, divisor: float) -> None:
    gradient = _gradient()
    unscaled = _combined(gradient, CAGrad[Params](SGD(LR), loss_ids=LOSS_IDS, rescale="none"))
    scaled = _combined(gradient, CAGrad[Params](SGD(LR), loss_ids=LOSS_IDS, rescale=rescale))  # ty: ignore[invalid-argument-type]
    chex.assert_trees_all_close(scaled, unscaled / divisor)


def test_zero_gradients_stay_finite() -> None:
    gradient = tree.map(jnp.zeros_like, _gradient())
    chex.assert_tree_all_finite(_combined(gradient, CAGrad[Params](SGD(LR), loss_ids=LOSS_IDS)))


def test_one_zero_gradient_stays_finite() -> None:
    gradient = _gradient()
    gradient["third"] = tree.map(jnp.zeros_like, gradient["third"])
    chex.assert_tree_all_finite(_combined(gradient, CAGrad[Params](SGD(LR), loss_ids=LOSS_IDS)))


def test_c_does_not_recompile() -> None:
    traces = 0
    gradient = _gradient()
    params = _params()

    @jax.jit
    def step(tx: CAGrad[Params]) -> Params:
        nonlocal traces
        traces += 1
        return tx.update(gradient, tx.init(params), params)[0]

    step(CAGrad[Params](SGD(LR), loss_ids=LOSS_IDS, c=0.2))
    step(CAGrad[Params](SGD(LR), loss_ids=LOSS_IDS, c=0.8))
    assert traces == 1


@pytest.mark.parametrize("scale", [1e-6, 1e-3, 1.0, 1e3])
@pytest.mark.parametrize("factor", [1.0, 2.0, -1.0])
def test_parallel_gradients_stay_finite_in_single_precision(scale: float, factor: float) -> None:
    base = jr.normal(jr.key(0), (5,), jnp.float32)
    gradient = {"first": {"w": scale * base}, "second": {"w": factor * scale * base}}
    params = {"w": jnp.zeros(5, jnp.float32)}
    tx = CAGrad[Params](SGD(LR), loss_ids=tuple(gradient))
    updates, _ = tx.update(gradient, tx.init(params), params)
    chex.assert_tree_all_finite(updates)
