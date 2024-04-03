"""Tests from optax._src.transform_test."""
from collections.abc import Callable
from typing import Any

import chex
import jax.numpy as jnp
import numpy as np
import pytest
from jax import tree
from optax import apply_updates

from tjax import RealArray
from tjax.gradient import (SGD, AddDecayedWeights, ApplyEvery, Centralize,
                           ChainedGradientTransformation, Ema, GradientTransformation, Scale,
                           ScaleByAdam, ScaleByParamBlockNorm, ScaleByParamBlockRMS, ScaleByRms,
                           ScaleByStddev, ScaleByTrustRatio, Trace)

STEPS = 50
LR = 1e-2
init_params = (np.asarray([1., 2.]), np.asarray([3., 4.]))
per_step_updates = (np.asarray([500., 5.]), np.asarray([300., 3.]))


def variant(x: Any) -> Any:
    return x


@pytest.mark.parametrize("constructor", [ScaleByAdam, ScaleByRms, ScaleByStddev, ScaleByTrustRatio,
                                         ScaleByParamBlockNorm, ScaleByParamBlockRMS])
def test_scalers(constructor: Callable[[], GradientTransformation[Any, Any]]) -> None:
    params = init_params

    scaler = constructor()
    init_fn = variant(scaler.init)
    transform_fn = variant(scaler.update)

    state = init_fn(params)
    chex.assert_tree_all_finite(state)

    updates, state = transform_fn(per_step_updates, state, params)
    chex.assert_tree_all_finite((params, updates, state))
    tree.map(lambda *args: chex.assert_equal_shape(args), params, updates)


def test_add_decayed_weights() -> None:
    # Define a transform that add decayed weights.
    # We can define a mask either as a pytree, or as a function that
    # returns the pytree. Below we define the pytree directly.
    mask = (True, {'a': True, 'b': False})
    tx = AddDecayedWeights[Any](0.1, mask=mask)
    # Define input updates and weights.
    updates = (
        jnp.zeros((2,), dtype=jnp.float32),
        {'a': jnp.zeros((2,), dtype=jnp.float32),
         'b': jnp.zeros((2,), dtype=jnp.float32)})
    weights = (
        jnp.ones((2,), dtype=jnp.float32),
        {'a': jnp.ones((2,), dtype=jnp.float32),
         'b': jnp.ones((2,), dtype=jnp.float32)})
    # This mask means that we will add decayed weights to the first two
    # terms in the input updates, but not to the last element.
    expected_tx_updates = (
        0.1 * jnp.ones((2,), dtype=jnp.float32),
        {'a': 0.1 * jnp.ones((2,), dtype=jnp.float32),
         'b': jnp.zeros((2,), dtype=jnp.float32)})
    # Apply transform
    state = tx.init(weights)
    transform_fn = variant(tx.update)
    new_updates, _ = transform_fn(updates, state, weights)
    # Assert output as expected.
    chex.assert_trees_all_close(new_updates, expected_tx_updates)


def test_ema() -> None:
    values = jnp.array([5.0, 7.0])
    decay = 0.9
    d = decay

    ema = Ema[Any](decay=decay, debias=False)
    state = ema.init(values[0])  # init to zeroes

    transform_fn = variant(ema.update)
    mean, state = transform_fn(values[0], state)
    np.testing.assert_allclose(mean, (1 - d) * values[0], atol=1e-4)

    mean, state = transform_fn(values[1], state)
    np.testing.assert_allclose(mean, (1 - d) * (values[1] + d * values[0]), atol=1e-2)


def test_ema_debias() -> None:
    values = jnp.array([5.0, 7.0])
    decay = 0.9
    d = decay

    ema = Ema[Any](decay=decay)
    state = ema.init(values[0])

    transform_fn = variant(ema.update)
    mean, state = transform_fn(values[0], state)
    np.testing.assert_allclose(mean, values[0], atol=1e-4)

    mean, state = transform_fn(values[1], state)
    np.testing.assert_allclose(mean, ((1 - d) * values[1] + d * (1 - d) * values[0]) / (1 - d**2),
                               atol=1e-2)
    # The state must not be debiased.
    np.testing.assert_allclose(state.data.ema,
                               (1 - d) * values[1] + d * (1 - d) * values[0],
                               atol=1e-2)


def test_apply_every() -> None:
    # The frequency of the application of SGD
    k = 4
    zero_update = (jnp.array([0., 0.]), jnp.array([0., 0.]))

    # optax SGD
    optax_sgd_params = init_params
    sgd_transform = SGD[tuple[RealArray, RealArray]](LR, 0.0)
    state_sgd = sgd_transform.init(optax_sgd_params)

    # optax SGD plus apply every
    optax_sgd_apply_every_params = init_params
    sgd_apply_every = ChainedGradientTransformation[tuple[RealArray, RealArray]](
        [ApplyEvery[Any](k=k),
         Trace[Any](decay=0, nesterov=False),
         Scale[Any](-LR)])
    state_sgd_apply_every = sgd_apply_every.init(optax_sgd_apply_every_params)
    transform_fn = variant(sgd_apply_every.update)

    for i in range(STEPS):
        # Apply a step of SGD
        updates_sgd, state_sgd = sgd_transform.update(per_step_updates, state_sgd, None)
        optax_sgd_params = apply_updates(optax_sgd_params, updates_sgd)

        # Apply a step of sgd_apply_every
        updates_sgd_apply_every, state_sgd_apply_every = transform_fn(
            per_step_updates, state_sgd_apply_every, None)
        optax_sgd_apply_every_params = apply_updates(
            optax_sgd_apply_every_params, updates_sgd_apply_every)

        # Every k steps, check equivalence.
        if i % k == k - 1:
            chex.assert_trees_all_close(
                optax_sgd_apply_every_params, optax_sgd_params,
                atol=1e-6, rtol=1e-5)
        # Otherwise, check update is zero.
        else:
            chex.assert_trees_all_close(
                updates_sgd_apply_every, zero_update, atol=0.0, rtol=0.0)


def test_scale() -> None:
    updates = per_step_updates
    for i in range(1, STEPS + 1):
        factor = 0.1 ** i
        rescaler = Scale[Any](factor)
        empty_state = rescaler.init(updates)
        # Apply rescaling.
        scaled_updates, _ = rescaler.update(updates, empty_state, updates)

        # Manually scale updates.
        def rescale(t: Any, *, factor: float = factor) -> Any:
            return t * factor
        manual_updates = tree.map(rescale, updates)
        # Check the rescaled updates match.
        chex.assert_trees_all_close(scaled_updates, manual_updates)


@pytest.mark.parametrize(("inputs", "outputs"), [
    ([1.0, 2.0], [1.0, 2.0]),
    ([[1.0, 2.0], [3.0, 4.0]], [[-0.5, 0.5], [-0.5, 0.5]]),
    ([[[1., 2.], [3., 4.]],
      [[5., 6.], [7., 8.]]], [[[-1.5, -0.5], [0.5, 1.5]], [[-1.5, -0.5], [0.5, 1.5]]]),
])
def test_centralize(inputs: Any, outputs: Any) -> None:
    inputs = jnp.asarray(inputs)
    outputs = jnp.asarray(outputs)
    centralizer = Centralize[Any]()
    empty_state = centralizer.init(inputs)
    centralized_inputs, _ = centralizer.update(inputs, empty_state, None)
    chex.assert_trees_all_close(centralized_inputs, outputs)
