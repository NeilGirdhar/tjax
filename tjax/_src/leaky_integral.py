from __future__ import annotations

from collections.abc import Callable
from typing import overload

import jax.numpy as jnp
import numpy as np
from array_api_compat import get_namespace
from jax import Array
from jax.dtypes import canonicalize_dtype
from jax.lax import scan
from jax.random import normal

from .annotations import (ComplexArray, IntegralArray, JaxComplexArray, JaxIntegralArray,
                          JaxRealArray, KeyArray, RealArray)
from .dataclasses.dataclass import dataclass


@overload
def leaky_integrate(value: JaxComplexArray,
                    time_step: JaxRealArray,
                    drift: JaxComplexArray | None = None,
                    decay: JaxComplexArray | None = None,
                    *,
                    leaky_average: bool = False
                    ) -> JaxComplexArray:
    ...


@overload
def leaky_integrate(value: ComplexArray,
                    time_step: RealArray,
                    drift: ComplexArray | None = None,
                    decay: ComplexArray | None = None,
                    *,
                    leaky_average: bool = False
                    ) -> ComplexArray:
    ...


def leaky_integrate(value: ComplexArray,
                    time_step: RealArray,
                    drift: ComplexArray | None = None,
                    decay: ComplexArray | None = None,
                    *,
                    leaky_average: bool = False
                    ) -> ComplexArray:
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
            return value
        xp = get_namespace(decay, time_step, value)
        return xp.exp(-decay * time_step) * value

    if decay is None:
        if leaky_average:
            raise ValueError
        xp = get_namespace(drift, time_step, value)
        return xp.asarray(value + drift * time_step)

    xp = get_namespace(drift, decay, time_step, value)
    scaled_integrand = (drift / decay) * -xp.expm1(-decay * time_step)

    if leaky_average:
        scaled_integrand *= decay.real

    return xp.exp(-decay * time_step) * value + scaled_integrand


@overload
def leaky_data_weight(iterations_times_time_step: JaxRealArray | JaxIntegralArray,
                      decay: JaxRealArray
                      ) -> JaxRealArray:
    ...


@overload
def leaky_data_weight(iterations_times_time_step: RealArray | IntegralArray,
                      decay: RealArray
                      ) -> RealArray:
    ...


def leaky_data_weight(iterations_times_time_step: RealArray | IntegralArray,
                      decay: RealArray
                      ) -> RealArray:
    """The amount of data that has been incorporated and has not been decayed.

    This equals leaky_integrate(0.0, iterations_times_time_step, 1.0, decay, leaky_average=True).
    """
    xp = get_namespace(iterations_times_time_step, decay)
    return -xp.expm1(-iterations_times_time_step * decay)


def diffused_leaky_integrate(value: JaxComplexArray,
                             time_step: JaxRealArray,
                             rng: KeyArray,
                             diffusion: JaxRealArray,
                             drift: JaxComplexArray | None = None,
                             decay: JaxComplexArray | None = None,
                             *,
                             leaky_average: bool = False
                             ) -> ComplexArray:
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


@dataclass
class _FilterCarry:
    iterations: JaxRealArray
    value: JaxComplexArray


@overload
def leaky_integrate_time_series(time_series: IntegralArray | RealArray,
                                decay: RealArray
                                ) -> RealArray: ...
@overload
def leaky_integrate_time_series(time_series: IntegralArray | ComplexArray,
                                decay: ComplexArray
                                ) -> ComplexArray: ...
def leaky_integrate_time_series(time_series: IntegralArray | ComplexArray,
                                decay: ComplexArray
                                ) -> ComplexArray:
    xp = get_namespace(time_series, decay)
    dtype = jnp.promote_types(canonicalize_dtype(time_series.dtype), jnp.float32)
    time_series = jnp.asarray(time_series, dtype=dtype)
    decay = jnp.asarray(decay)
    if issubclass(time_series.dtype.type, np.integer):
        msg = "Cast the time series to a floating type."
        raise TypeError(msg)

    def g(carry: _FilterCarry, drift: JaxComplexArray) -> tuple[_FilterCarry, JaxComplexArray]:
        new_iterations = carry.iterations + 1.0
        data_weight = leaky_data_weight(new_iterations, decay.real)
        new_value = leaky_integrate(carry.value, jnp.asarray(1.0), drift, decay, leaky_average=True)
        assert isinstance(new_value, Array)
        new_carry = _FilterCarry(new_iterations, new_value)
        outputted_value = new_value / data_weight
        return new_carry, outputted_value

    # Cast the dtype from integer to floating point to prevent integer rounding.
    initial_value = jnp.zeros(time_series[0].shape, dtype=time_series.dtype)
    initial_carry = _FilterCarry(jnp.asarray(0.0), initial_value)

    _, filtered_time_series = scan(g, initial_carry, time_series)
    return xp.asarray(filtered_time_series)


def leaky_covariance(x_time_series: ComplexArray,
                     y_time_series: ComplexArray,
                     decay: ComplexArray,
                     *,
                     covariance_matrix: bool = False) -> ComplexArray:
    get_namespace(x_time_series, y_time_series, decay)  # Verify namespace.
    times: Callable[[ComplexArray, ComplexArray], ComplexArray]
    if covariance_matrix:
        if x_time_series.shape[0] != y_time_series.shape[0]:
            raise ValueError
        s = (np.newaxis,)

        def times(a: ComplexArray, b: ComplexArray, /) -> ComplexArray:
            a_selected = a[..., *(s * (b.ndim - 1))]
            b_selected = b[:, *(s * (a.ndim - 1))]
            return a_selected * b_selected
    else:
        if x_time_series.shape != y_time_series.shape:
            raise ValueError

        def times(a: ComplexArray, b: ComplexArray, /) -> ComplexArray:
            return a * b
    x = leaky_integrate_time_series(x_time_series, decay)
    y = leaky_integrate_time_series(y_time_series, decay)
    xy = leaky_integrate_time_series(times(x_time_series, y_time_series), decay)
    return xy - times(x, y)
