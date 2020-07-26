from __future__ import annotations

from typing import Optional, Tuple

from jax import numpy as jnp

from tjax import Generator, Tensor, real_dtype

__all__ = ['leaky_integrate', 'diffused_leaky_integrate']


def leaky_integrate(value: Tensor,
                    time_step: real_dtype,
                    drift: Optional[Tensor] = None,
                    decay: Optional[Tensor] = None,
                    *,
                    leaky_average: bool = False) -> Tensor:
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


def diffused_leaky_integrate(value: Tensor,
                             time_step: real_dtype,
                             rng: Generator,
                             diffusion: Tensor,
                             drift: Optional[Tensor] = None,
                             decay: Optional[Tensor] = None,
                             *,
                             leaky_average: bool = False) -> Tuple[Tensor, Generator]:
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
    new_rng, jump = rng.normal(jnp.sqrt(variance), shape=value.shape)
    return (leaky_integrate(value, time_step, drift, decay, leaky_average=leaky_average) + jump,
            new_rng)
