"""Compare the multi-loss transformations on the Helmholtz benchmark of Wang et al.

This produces the results in ``docs/multi-loss-benchmark.md``.  A physics-informed network balances
a PDE residual loss against a boundary loss.  Pass the number of steps and seeds as arguments, for
example ``python examples/multi_loss_benchmark.py 40000 5``.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import optax

import tjax.gradient as tg
from tjax import JaxArray, KeyArray, PyTree

LAYERS = (2, 50, 50, 50, 1)
BATCH = 128
A1, A2, K = 1.0, 4.0, 1.0
LOSS_IDS = ("residual", "boundary")

type Parameters = list[dict[str, JaxArray]]
type Method = Callable[[JaxArray], tg.MultiLossGradientTransformation[Any, Parameters]]


def exact(x: JaxArray, y: JaxArray) -> JaxArray:
    return jnp.sin(A1 * jnp.pi * x) * jnp.sin(A2 * jnp.pi * y)


def forcing(x: JaxArray, y: JaxArray) -> JaxArray:
    return (K**2 - (A1 * jnp.pi) ** 2 - (A2 * jnp.pi) ** 2) * exact(x, y)


def initialize(key: KeyArray) -> Parameters:
    return [
        {"w": jr.normal(k, (n, m)) * jnp.sqrt(2.0 / (n + m)), "b": jnp.zeros(m)}
        for k, n, m in zip(jr.split(key, len(LAYERS) - 1), LAYERS[:-1], LAYERS[1:], strict=True)
    ]


def network(parameters: Parameters, x: JaxArray, y: JaxArray) -> JaxArray:
    hidden = jnp.stack([x, y])
    for layer in parameters[:-1]:
        hidden = jnp.tanh(hidden @ layer["w"] + layer["b"])
    return (hidden @ parameters[-1]["w"] + parameters[-1]["b"])[0]


def residual(parameters: Parameters, x: JaxArray, y: JaxArray) -> JaxArray:
    u_xx = jax.grad(jax.grad(network, 1), 1)(parameters, x, y)
    u_yy = jax.grad(jax.grad(network, 2), 2)(parameters, x, y)
    return u_xx + u_yy + K**2 * network(parameters, x, y) - forcing(x, y)


def losses(parameters: Parameters, key: KeyArray) -> tuple[JaxArray, JaxArray]:
    interior_key, boundary_key = jr.split(key)
    x, y = jr.uniform(interior_key, (2, BATCH), minval=-1.0, maxval=1.0)
    residual_loss = jnp.mean(jax.vmap(residual, (None, 0, 0))(parameters, x, y) ** 2)
    edges = jr.uniform(boundary_key, (4, BATCH), minval=-1.0, maxval=1.0)
    ones = jnp.ones(BATCH)
    boundary_x = jnp.concatenate([edges[0], edges[1], -ones, ones])
    boundary_y = jnp.concatenate([-ones, ones, edges[2], edges[3]])
    predicted = jax.vmap(network, (None, 0, 0))(parameters, boundary_x, boundary_y)
    return residual_loss, jnp.mean((predicted - exact(boundary_x, boundary_y)) ** 2)


def relative_error(parameters: Parameters) -> JaxArray:
    grid = jnp.linspace(-1.0, 1.0, 101)
    x, y = (axis.ravel() for axis in jnp.meshgrid(grid, grid))
    difference = jax.vmap(network, (None, 0, 0))(parameters, x, y) - exact(x, y)
    return jnp.linalg.vector_norm(difference) / jnp.linalg.vector_norm(exact(x, y))


def gradients(parameters: Parameters, key: KeyArray) -> dict[str, Parameters]:
    return {
        loss_id: jax.grad(lambda p, index=index: losses(p, key)[index])(parameters)
        for index, loss_id in enumerate(LOSS_IDS)
    }


def train_with_sum(seed: JaxArray, schedule: tg.Schedule, steps: int) -> JaxArray:
    """Train on the sum of the losses, with no balancing."""
    transform = tg.Adam[Parameters](schedule)
    parameters = initialize(jr.key(seed))

    def step(carry: tuple[Parameters, PyTree], key: KeyArray) -> tuple[Any, None]:
        parameters, state = carry
        gradient = jax.grad(lambda p: sum(losses(p, key)))(parameters)
        updates, state = transform.update(gradient, state, parameters)
        return (optax.apply_updates(parameters, updates), state), None

    keys = jr.split(jr.fold_in(jr.key(seed), 1), steps)
    (parameters, _), _ = jax.lax.scan(step, (parameters, transform.init(parameters)), keys)
    return relative_error(parameters)


def train(seed: JaxArray, method: Method, steps: int) -> JaxArray:
    transform = method(seed)
    parameters = initialize(jr.key(seed))

    def step(carry: tuple[Parameters, PyTree], key: KeyArray) -> tuple[Any, None]:
        parameters, state = carry
        updates, state = transform.update(gradients(parameters, key), state, parameters)
        return (optax.apply_updates(parameters, updates), state), None

    keys = jr.split(jr.fold_in(jr.key(seed), 1), steps)
    initial = (parameters, transform.init(parameters))
    (parameters, _), _ = jax.lax.scan(step, initial, keys)
    return relative_error(parameters)


def report(name: str, errors: JaxArray, start: float) -> None:
    print(  # ruff: ignore[print]
        f"{name:30} mean {float(jnp.mean(errors)):.4f}  best {float(jnp.min(errors)):.4f}  "
        f"worst {float(jnp.max(errors)):.4f}  ({time.time() - start:.0f}s)",
        flush=True,
    )


def main() -> None:
    steps, n_seeds = int(sys.argv[1]), int(sys.argv[2])
    seeds = jnp.arange(n_seeds)
    schedule = tg.Schedule(optax.exponential_decay(1e-3, 1000, 0.9))
    adam = tg.Adam[Parameters](schedule)
    methods: dict[str, Method] = {
        "GradientStatisticsAnnealing": lambda _: tg.GradientStatisticsAnnealing(
            adam, loss_ids=LOSS_IDS
        ),
        "MTAdam": lambda _: tg.MTAdam(schedule, loss_ids=LOSS_IDS),
        "IMTLG": lambda _: tg.IMTLG(adam, loss_ids=LOSS_IDS),
        "PCGrad": lambda seed: tg.PCGrad(adam, jr.key(seed), loss_ids=LOSS_IDS),
        "GradVac": lambda seed: tg.GradVac(adam, jr.key(seed), loss_ids=LOSS_IDS),
        "CAGrad": lambda _: tg.CAGrad(adam, loss_ids=LOSS_IDS),
    }

    start = time.time()
    errors = jax.jit(jax.vmap(lambda seed: train_with_sum(seed, schedule, steps)))(seeds)
    report("No balancing", errors, start)
    for name, method in methods.items():
        start = time.time()
        errors = jax.jit(jax.vmap(lambda seed, method=method: train(seed, method, steps)))(seeds)
        report(name, errors, start)


if __name__ == "__main__":
    main()
