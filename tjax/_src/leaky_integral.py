from __future__ import annotations

from typing import Callable, Optional, Tuple, overload

import jax.numpy as jnp
import numpy as np
from jax.lax import scan
from jax.random import KeyArray, normal

from .annotations import ComplexArray, ComplexNumeric, RealArray, RealNumeric
from .dataclasses import dataclass

__all__ = ['leaky_integrate', 'diffused_leaky_integrate', 'leaky_data_weight',
           'leaky_integrate_time_series', 'leaky_covariance']


@overload
def leaky_integrate(value: RealArray,
                    time_step: RealNumeric,
                    drift: Optional[RealNumeric] = None,
                    decay: Optional[RealNumeric] = None,
                    *,
                    leaky_average: bool = False) -> RealArray:
    ...


@overload
def leaky_integrate(value: RealNumeric,  # type: ignore[misc]
                    time_step: RealNumeric,
                    drift: Optional[RealNumeric] = None,
                    decay: Optional[RealNumeric] = None,
                    *,
                    leaky_average: bool = False) -> RealNumeric:
    ...


@overload
def leaky_integrate(value: ComplexArray,
                    time_step: RealNumeric,
                    drift: Optional[ComplexNumeric] = None,
                    decay: Optional[ComplexNumeric] = None,
                    *,
                    leaky_average: bool = False) -> ComplexArray:
    ...


@overload
def leaky_integrate(value: ComplexNumeric,
                    time_step: RealNumeric,
                    drift: Optional[ComplexNumeric] = None,
                    decay: Optional[ComplexNumeric] = None,
                    *,
                    leaky_average: bool = False) -> ComplexNumeric:
    ...


def leaky_integrate(value: ComplexNumeric,
                    time_step: RealNumeric,
                    drift: Optional[ComplexNumeric] = None,
                    decay: Optional[ComplexNumeric] = None,
                    *,
                    leaky_average: bool = False) -> ComplexNumeric:
    """
    Update the value so that it is the leaky integral (or leaky average).
    Args:
        value: The current value of the leaky integral or average.
        time_step: The number of seconds that have passed.
        decay: If provided, must have positive real component, and the value decays by exp(-decay)
            every second.
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


@overload
def diffused_leaky_integrate(value: RealArray,
                             time_step: RealNumeric,
                             rng: KeyArray,
                             diffusion: RealNumeric,
                             drift: Optional[RealNumeric] = None,
                             decay: Optional[RealNumeric] = None,
                             *,
                             leaky_average: bool = False) -> RealArray:
    ...


@overload
def diffused_leaky_integrate(value: ComplexArray,
                             time_step: RealNumeric,
                             rng: KeyArray,
                             diffusion: RealNumeric,
                             drift: Optional[ComplexNumeric] = None,
                             decay: Optional[ComplexNumeric] = None,
                             *,
                             leaky_average: bool = False) -> ComplexArray:
    ...


def diffused_leaky_integrate(value: ComplexArray,
                             time_step: RealNumeric,
                             rng: KeyArray,
                             diffusion: RealNumeric,
                             drift: Optional[ComplexNumeric] = None,
                             decay: Optional[ComplexNumeric] = None,
                             *,
                             leaky_average: bool = False) -> ComplexArray:
    """
    Update an Ornstein-Uhlenbeck process.

    Args:
        value: The current value of the leaky integral or average.
        time_step: The number of seconds that have passed.
        rng: The key array for the stochastic process.
        diffusion: The diffusion for the stochastic process.
        decay: If provided, the value decays by exp(-decay) every second.
        drift: If provided, the value increases by this every second.
        leaky_average: A flag indicating a leaky average rather than a leaky integral.  This scales
            the drift by the real component (in case the decay is complex) of the decay.
    """
    variance = (diffusion * time_step
                if decay is None
                else diffusion / decay * -jnp.expm1(-decay * time_step))
    jump = jnp.sqrt(variance) * normal(rng, value.shape)
    return leaky_integrate(value, time_step, drift, decay, leaky_average=leaky_average) + jump


def leaky_data_weight(iterations_times_time_step: RealNumeric,
                      decay: RealNumeric) -> RealNumeric:
    """
    Returns: The amount of data that has been incorporated and has not been decayed.  That is,
        leaky_integrate(0.0, iterations_times_time_step, 1.0, decay, leaky_average=True)
    """
    return -jnp.expm1(-iterations_times_time_step * decay)


@dataclass
class _FilterCarry:
    iterations: RealNumeric
    value: ComplexArray


@overload
def leaky_integrate_time_series(time_series: RealArray, decay: RealNumeric) -> RealArray:
    ...


@overload
def leaky_integrate_time_series(time_series: ComplexArray, decay: ComplexNumeric) -> ComplexArray:
    ...


def leaky_integrate_time_series(time_series: ComplexArray, decay: ComplexNumeric) -> ComplexArray:
    if issubclass(time_series.dtype.type, np.integer):
        raise TypeError("Cast the time series to a floating type.")

    def g(carry: _FilterCarry, drift: ComplexNumeric) -> Tuple[_FilterCarry, ComplexArray]:
        new_iterations = carry.iterations + 1.0
        data_weight = leaky_data_weight(new_iterations, decay.real)
        new_value = leaky_integrate(carry.value, 1.0, drift, decay, leaky_average=True)
        new_carry = _FilterCarry(new_iterations, new_value)
        outputted_value = new_value / data_weight
        return new_carry, outputted_value

    # Cast the dtype from integer to floating point to prevent integer rounding.
    initial_value = np.zeros(time_series[0].shape, dtype=time_series.dtype)
    initial_carry = _FilterCarry(0.0, initial_value)

    _, filtered_time_series = scan(g, initial_carry, time_series)
    return filtered_time_series


@overload
def leaky_covariance(x_time_series: RealArray,
                     y_time_series: RealArray,
                     decay: RealNumeric,
                     covariance_matrix: bool = False) -> RealArray:
    ...


@overload
def leaky_covariance(x_time_series: ComplexArray,
                     y_time_series: ComplexArray,
                     decay: ComplexNumeric,
                     covariance_matrix: bool = False) -> ComplexArray:
    ...


def leaky_covariance(x_time_series: ComplexArray,
                     y_time_series: ComplexArray,
                     decay: ComplexNumeric,
                     covariance_matrix: bool = False) -> ComplexArray:
    times: Callable[[ComplexArray, ComplexArray], ComplexArray]
    if covariance_matrix:
        if x_time_series.shape[0] != y_time_series.shape[0]:
            raise ValueError
        s = (np.newaxis,)

        def times(a: ComplexArray, b: ComplexArray, /) -> ComplexArray:
            return a[(..., *(s * (b.ndim - 1)))] * b[(slice(None), *(s * (a.ndim - 1)))]
    else:
        if x_time_series.shape != y_time_series.shape:
            raise ValueError

        def times(a: ComplexArray, b: ComplexArray, /) -> ComplexArray:
            return a * b
    x = leaky_integrate_time_series(x_time_series, decay)
    y = leaky_integrate_time_series(y_time_series, decay)
    xy = leaky_integrate_time_series(times(x_time_series, y_time_series), decay)
    return xy - times(x, y)
