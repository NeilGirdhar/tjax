from __future__ import annotations

from typing import Callable, overload

import jax.numpy as jnp
import numpy as np
from jax.lax import scan
from jax.random import KeyArray, normal

from .annotations import (ComplexArray, ComplexNumeric, JaxComplexArray, JaxRealArray, RealArray,
                          RealNumeric)
from .dataclasses import dataclass

__all__ = ['leaky_integrate', 'diffused_leaky_integrate', 'leaky_data_weight',
           'leaky_integrate_time_series', 'leaky_covariance']


@overload
def leaky_integrate(value: RealNumeric,
                    time_step: RealNumeric,
                    drift: RealNumeric | None = None,
                    decay: RealNumeric | None = None,
                    *,
                    leaky_average: bool = False) -> JaxRealArray:
    ...


@overload
def leaky_integrate(value: ComplexNumeric,
                    time_step: RealNumeric,
                    drift: ComplexNumeric | None = None,
                    decay: ComplexNumeric | None = None,
                    *,
                    leaky_average: bool = False) -> JaxComplexArray:
    ...


def leaky_integrate(value: ComplexNumeric,
                    time_step: RealNumeric,
                    drift: ComplexNumeric | None = None,
                    decay: ComplexNumeric | None = None,
                    *,
                    leaky_average: bool = False) -> JaxComplexArray:
    """Update the value so that it is the leaky integral (or leaky average).

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
            return jnp.asarray(value)
        return jnp.exp(-decay * time_step) * value

    if decay is None:
        if leaky_average:
            raise ValueError
        return jnp.asarray(value + drift * time_step)

    scaled_integrand = (drift / decay) * -jnp.expm1(-decay * time_step)

    if leaky_average:
        scaled_integrand *= decay.real

    return jnp.exp(-decay * time_step) * value + scaled_integrand


@overload
def diffused_leaky_integrate(value: RealArray,
                             time_step: RealNumeric,
                             rng: KeyArray,
                             diffusion: RealNumeric,
                             drift: RealNumeric | None = None,
                             decay: RealNumeric | None = None,
                             *,
                             leaky_average: bool = False) -> JaxRealArray:
    ...


@overload
def diffused_leaky_integrate(value: ComplexArray,
                             time_step: RealNumeric,
                             rng: KeyArray,
                             diffusion: RealNumeric,
                             drift: ComplexNumeric | None = None,
                             decay: ComplexNumeric | None = None,
                             *,
                             leaky_average: bool = False) -> JaxComplexArray:
    ...


def diffused_leaky_integrate(value: ComplexArray,
                             time_step: RealNumeric,
                             rng: KeyArray,
                             diffusion: RealNumeric,
                             drift: ComplexNumeric | None = None,
                             decay: ComplexNumeric | None = None,
                             *,
                             leaky_average: bool = False) -> JaxComplexArray:
    """Update an Ornstein-Uhlenbeck process.

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
                      decay: RealNumeric) -> JaxRealArray:
    """The amount of data that has been incorporated and has not been decayed.

    This equals leaky_integrate(0.0, iterations_times_time_step, 1.0, decay, leaky_average=True).
    """
    return -jnp.expm1(-iterations_times_time_step * decay)


@dataclass
class _FilterCarry:
    iterations: JaxRealArray
    value: JaxComplexArray


@overload
def leaky_integrate_time_series(time_series: RealArray, decay: RealNumeric
                                ) -> JaxRealArray:
    ...


@overload
def leaky_integrate_time_series(time_series: ComplexArray, decay: ComplexNumeric
                                ) -> JaxComplexArray:
    ...


def leaky_integrate_time_series(time_series: ComplexArray, decay: ComplexNumeric
                                ) -> JaxComplexArray:
    if issubclass(time_series.dtype.type, np.integer):
        msg = "Cast the time series to a floating type."
        raise TypeError(msg)

    def g(carry: _FilterCarry, drift: ComplexNumeric) -> tuple[_FilterCarry, JaxComplexArray]:
        new_iterations = carry.iterations + 1.0
        data_weight = leaky_data_weight(new_iterations, decay.real)
        new_value = leaky_integrate(carry.value, 1.0, drift, decay, leaky_average=True)
        new_carry = _FilterCarry(new_iterations, new_value)
        outputted_value = new_value / data_weight
        return new_carry, outputted_value

    # Cast the dtype from integer to floating point to prevent integer rounding.
    initial_value = jnp.zeros(time_series[0].shape, dtype=time_series.dtype)
    initial_carry = _FilterCarry(jnp.asarray(0.0), initial_value)

    _, filtered_time_series = scan(g, initial_carry, time_series)
    return filtered_time_series


@overload
def leaky_covariance(x_time_series: RealArray,
                     y_time_series: RealArray,
                     decay: RealNumeric,
                     *,
                     covariance_matrix: bool = False) -> JaxRealArray:
    ...


@overload
def leaky_covariance(x_time_series: ComplexArray,
                     y_time_series: ComplexArray,
                     decay: ComplexNumeric,
                     *,
                     covariance_matrix: bool = False) -> JaxComplexArray:
    ...


def leaky_covariance(x_time_series: ComplexArray,
                     y_time_series: ComplexArray,
                     decay: ComplexNumeric,
                     *,
                     covariance_matrix: bool = False) -> JaxComplexArray:
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
