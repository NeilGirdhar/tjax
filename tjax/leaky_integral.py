from __future__ import annotations

from numbers import Integral
from typing import Optional, Tuple

import jax.numpy as jnp
import numpy as np
from chex import Array
from jax.lax import scan

from .dataclass import dataclass
from .dtypes import real_dtype
from .generator import Generator

__all__ = ['leaky_integrate', 'diffused_leaky_integrate', 'leaky_data_weight',
           'leaky_integrate_time_series', 'leaky_covariance']


def leaky_integrate(value: Array,
                    time_step: Array,
                    drift: Optional[Array] = None,
                    decay: Optional[Array] = None,
                    *,
                    leaky_average: bool = False) -> Array:
    """
    Update the value so that it is the leaky integral (or leaky average).
    Args:
        value: The current value of the leaky integral or average.
        time_step: The number of seconds that have passed.
        decay: If provided, must be positive, and the value decays by exp(-decay) every second.
        drift: If provided, the value increases by this every second.
        leaky_average: A flag indicating a leaky average rather than a leaky integral.  This scales
            the drift by the real component (in case the decay is complex) of the decay.
    """
    if drift is None:
        if decay is None:
            return value
        return value * jnp.exp(-decay * time_step)

    if decay is None:
        if leaky_average:
            raise ValueError
        return value + drift * time_step

    scaled_integrand = (drift / decay) * -jnp.expm1(-decay * time_step)

    if leaky_average:
        scaled_integrand *= decay.real

    return value * jnp.exp(-decay * time_step) + scaled_integrand


def diffused_leaky_integrate(value: Array,
                             time_step: Array,
                             rng: Generator,
                             diffusion: Array,
                             drift: Optional[Array] = None,
                             decay: Optional[Array] = None,
                             *,
                             leaky_average: bool = False) -> Tuple[Array, Generator]:
    """
    Update an Ornstein-Uhlenbeck process.

    Args:
        value: The current value of the leaky integral or average.
        time_step: The number of seconds that have passed.
        rng: The random number generator for the stochastic process.
        diffusion: The diffusion for the stochastic process.
        decay: If provided, the value decays by exp(-decay) every second.
        drift: If provided, the value increases by this every second.
        leaky_average: A flag indicating a leaky average rather than a leaky integral.  This scales
            the drift by the real component (in case the decay is complex) of the decay.
    """
    variance = (diffusion * time_step
                if decay is None
                else diffusion / decay * -jnp.expm1(-decay * time_step))
    jump, new_rng = rng.normal(jnp.sqrt(variance), shape=value.shape)
    return (leaky_integrate(value, time_step, drift, decay, leaky_average=leaky_average) + jump,
            new_rng)


def leaky_data_weight(iterations_times_time_step: Array,
                      decay: Array) -> Array:
    """
    Returns: The amount of data that has been incorporated and has not been decayed.
    """
    return leaky_integrate(0.0, iterations_times_time_step, 1.0, decay, leaky_average=True)


@dataclass
class _FilterCarry:
    iterations: Array
    value: Array


def leaky_integrate_time_series(time_series: Array, decay: Array) -> Array:
    """
    Args:
        time_series: A sequence of
        f: A function that maps from value, drift to new_value.
        reweighted: Rescales the early
    """
    def g(carry: _FilterCarry, drift: Array) -> Tuple[_FilterCarry, Array]:
        new_iterations = carry.iterations + 1.0
        data_weight = leaky_data_weight(new_iterations, decay)

        new_value = leaky_integrate(carry.value, 1.0, drift, decay, leaky_average=True)
        new_carry = _FilterCarry(new_iterations, new_value)
        outputted_value = new_value / data_weight
        return new_carry, outputted_value

    # Cast the dtype from integer to floating point to prevent integer rounding.
    initial_value = jnp.zeros(time_series[0].shape,
                              dtype=(real_dtype
                                     if issubclass(time_series.dtype.type, Integral)
                                     else time_series.dtype))
    initial_carry = _FilterCarry(0.0, initial_value)

    _, filtered_time_series = scan(g, initial_carry, time_series)
    return filtered_time_series


def leaky_covariance(x_time_series: Array,
                     y_time_series: Array,
                     decay: Array,
                     covariance_matrix: bool = False) -> Array:
    if covariance_matrix:
        if x_time_series.shape[0] != y_time_series.shape[0]:
            raise ValueError
        s = (np.newaxis,)

        def times(a: Array, b: Array) -> Array:
            return a[(..., *(s * (b.ndim - 1)))] * b[(slice(None), *(s * (a.ndim - 1)))]
    else:
        if x_time_series.shape != y_time_series.shape:
            raise ValueError
        times = jnp.multiply
    x = leaky_integrate_time_series(x_time_series, decay)
    y = leaky_integrate_time_series(y_time_series, decay)
    xy = leaky_integrate_time_series(times(x_time_series, y_time_series), decay)
    return xy - times(x, y)
