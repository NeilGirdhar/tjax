from __future__ import annotations

from typing import overload

import jax.numpy as jnp
import jax.random as jr
from array_api_compat import array_namespace

from .annotations import (
    ComplexArray,
    IntegralArray,
    JaxComplexArray,
    JaxIntegralArray,
    JaxRealArray,
    KeyArray,
    RealArray,
    RealNumeric,
)

_SERIES_THRESHOLD = 1e-3


def _step_weight(x: ComplexArray) -> ComplexArray:
    """Return (1 - exp(-x)) / x, which equals 1 at x = 0.

    This is the fraction of a constant input over a step that survives a decay of x over that step.
    The quotient is 0/0 at zero, so a Taylor series replaces it near zero.  The series's first
    omitted term, x^4 / 120, is below 1e-14 under the threshold, so the series is accurate in single
    and double precision.  The quotient's input is replaced near zero so that its unused gradient
    stays finite.
    """
    xp = array_namespace(x)
    small = xp.abs(x) < _SERIES_THRESHOLD
    safe_x = xp.where(small, xp.ones_like(x), x)
    series = 1 - x / 2 * (1 - x / 3 * (1 - x / 4))
    return xp.where(small, series, -xp.expm1(-safe_x) / safe_x)


@overload
def leaky_integrate(
    value: JaxComplexArray,
    time_step: JaxRealArray | float,
    drift: JaxComplexArray | None = None,
    decay: JaxComplexArray | None = None,
    *,
    leaky_average: bool = False,
) -> JaxComplexArray: ...


@overload
def leaky_integrate(
    value: ComplexArray,
    time_step: RealNumeric,
    drift: ComplexArray | None = None,
    decay: ComplexArray | None = None,
    *,
    leaky_average: bool = False,
) -> ComplexArray: ...


def leaky_integrate(
    value: ComplexArray,
    time_step: RealNumeric,
    drift: ComplexArray | None = None,
    decay: ComplexArray | None = None,
    *,
    leaky_average: bool = False,
) -> ComplexArray:
    """Update the value so that it is the leaky integral (or leaky average).

    Args:
        value: The current value of the leaky integral or average.
        time_step: The number of seconds that have passed.
        decay: If provided, must have nonnegative real component, and the value decays by
            exp(-decay) every second.
        drift: If provided, the value increases by this every second.
        leaky_average: A flag indicating a leaky average rather than a leaky integral.  This scales
            the drift by the real component (in case the decay is complex) of the decay.
    """
    if drift is None:
        if decay is None:
            return value
        xp = array_namespace(decay, time_step, value)
        return xp.exp(-decay * time_step) * value

    if decay is None:
        if leaky_average:
            raise ValueError
        xp = array_namespace(drift, time_step, value)
        return xp.asarray(value + drift * time_step)

    xp = array_namespace(drift, decay, time_step, value)
    scaled_integrand = drift * time_step * _step_weight(decay * time_step)

    if leaky_average:
        scaled_integrand *= decay.real

    return xp.exp(-decay * time_step) * value + scaled_integrand


@overload
def leaky_data_weight(
    iterations_times_time_step: JaxRealArray | JaxIntegralArray, decay: JaxRealArray
) -> JaxRealArray: ...


@overload
def leaky_data_weight(
    iterations_times_time_step: RealArray | IntegralArray, decay: RealArray
) -> RealArray: ...


def leaky_data_weight(
    iterations_times_time_step: RealArray | IntegralArray, decay: RealArray
) -> RealArray:
    """The amount of data that has been incorporated and has not been decayed.

    This equals leaky_integrate(0.0, iterations_times_time_step, 1.0, decay, leaky_average=True).
    """
    xp = array_namespace(iterations_times_time_step, decay)
    return -xp.expm1(-iterations_times_time_step * decay)


def diffused_leaky_integrate(
    value: JaxComplexArray,
    time_step: JaxRealArray,
    rng: KeyArray,
    diffusion: JaxRealArray,
    drift: JaxComplexArray | None = None,
    decay: JaxComplexArray | None = None,
    *,
    leaky_average: bool = False,
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
    variance = (
        diffusion * time_step
        if decay is None
        else diffusion * time_step * _step_weight(decay * time_step)
    )
    jump = jnp.sqrt(variance) * jr.normal(rng, value.shape)
    return leaky_integrate(value, time_step, drift, decay, leaky_average=leaky_average) + jump
