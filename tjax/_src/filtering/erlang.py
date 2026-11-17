"""Chains of leaky integrals that share one decay rate."""

from __future__ import annotations

import math
from typing import Self, override

import jax.numpy as jnp
import numpy as np
from jax import tree

from tjax._src.annotations import ComplexNumeric, JaxComplexArray, PyTree, RealNumeric
from tjax.dataclasses import dataclass

from .filter import StagedFilter, mix_stages

_SERIES_TERMS = 48


def _powers(base: JaxComplexArray, count: int) -> JaxComplexArray:
    """Return base^0, ..., base^(count - 1).

    A cumulative product keeps the gradient finite at zero, which a power with an array exponent
    does not.
    """
    factors = jnp.concatenate([jnp.ones((1,), dtype=base.dtype), jnp.full((count - 1,), base)])
    return jnp.cumprod(factors)


def _erlang_input_weights(x: JaxComplexArray, order: int) -> JaxComplexArray:
    """Return (1 - exp(-x) sum_(m < j) x^m / m!) / x^j for j = 1, ..., order.

    The jth entry is the integral of exp(-x u) u^(j - 1) / (j - 1)! over u from 0 to 1, which is
    1 / j! at x = 0.

    Below |x| = order + 4, the closed form cancels badly, so the series
    exp(-x) sum_(m >= 0) x^m / (j + m)! replaces it.  Its terms are positive for real x, and 48
    terms reach double precision for orders up to 8.  Above that threshold, the subtracted sum is
    small, so the closed form is accurate.  Each branch's input is replaced where it is unused, so
    that the gradient stays finite.
    """
    threshold = order + 4
    small = jnp.abs(x) < threshold
    j = np.arange(1, order + 1)

    series_x = jnp.where(small, x, jnp.zeros_like(x))
    inverse_factorials = np.asarray(
        [[1 / math.factorial(a + m) for m in range(_SERIES_TERMS)] for a in j]
    )
    series = jnp.exp(-series_x) * (inverse_factorials @ _powers(series_x, _SERIES_TERMS))

    closed_x = jnp.where(small, jnp.full_like(x, threshold), x)
    powers = _powers(closed_x, order + 1)
    partial_sums = jnp.cumsum(powers[:-1] / np.asarray([math.factorial(m) for m in range(order)]))
    closed = (1 - jnp.exp(-closed_x) * partial_sums) / powers[1:]
    return jnp.where(small, series, closed)


def _lower_triangle(order: int) -> np.ndarray:
    return np.tril(np.ones((order, order), dtype=bool))


@dataclass
class ErlangChain(StagedFilter):
    """A chain of leaky integrals that share one decay rate, read out through stage weights.

    Each leaf's state has shape (order, *s), where order is the length of the weights.  The kernel
    is psi(u) = sum_j weights_j u^(j - 1) exp(-decay u) / (j - 1)!.  Stage j alone peaks
    (j - 1) / decay seconds after the signal, so the weights place the kernel's mass at a delay.
    The kernel is nonnegative when the weights are nonnegative.

    With one stage and a weight of one, this is the leaky integral.

    Args:
        decay: The shared decay rate, with shape ().  Its real component must be nonnegative.
        weights: The weights of the stages, with shape (order,).
    """

    decay: JaxComplexArray
    weights: JaxComplexArray

    @classmethod
    def soft_window(cls, decay: ComplexNumeric, order: int) -> Self:
        """Return a chain whose kernel weights the lags up to about order / decay seconds equally.

        With weights_j = decay^(j - 1), the kernel is the probability that a Poisson variable with
        mean decay * u is at most order - 1.  It is near one at short lags and falls through one
        half near order / decay seconds.
        """
        decay = jnp.asarray(decay)
        return cls(decay, _powers(decay, order))

    @property
    @override
    def order(self) -> int:
        return self.weights.shape[0]

    @property
    @override
    def readout(self) -> JaxComplexArray:
        return self.weights

    @property
    @override
    def transition_pattern(self) -> np.ndarray:
        """The lower triangle, since each stage feeds only later ones.

        This override is only for speed: each step skips the zeros above the diagonal.
        """
        return _lower_triangle(self.order)

    @override
    def discretize(self, time_step: RealNumeric) -> tuple[JaxComplexArray, JaxComplexArray]:
        """Return the exact step of the chain with the signal held over the step.

        The chain is ds_1/dt = -decay s_1 + c and ds_j/dt = -decay s_j + s_(j - 1).
        """
        order = self.order
        decay = jnp.asarray(self.decay)
        time_step = jnp.asarray(time_step, dtype=jnp.result_type(decay.real.dtype, time_step))
        x = decay * time_step
        lag = np.subtract.outer(np.arange(order), np.arange(order))
        clipped_lag = np.maximum(lag, 0)
        inverse_factorial = np.where(
            lag >= 0, np.vectorize(lambda m: 1 / math.factorial(m))(clipped_lag), 0.0
        )
        step_powers = _powers(time_step, order + 1)
        transition = jnp.exp(-x) * step_powers[clipped_lag] * inverse_factorial
        input_weight = step_powers[1:] * _erlang_input_weights(x, order)
        return transition, input_weight

    def laguerre_matrix(self) -> JaxComplexArray:
        """Return the matrix that maps the stages to Laguerre components, with shape (order, order).

        Component m is the signal filtered by the Laguerre function
        sqrt(2 decay) exp(-decay u) L_m(2 decay u), where L_m is the mth Laguerre polynomial.  These
        kernels are orthonormal over continuous time, which conditions regressions on them well.
        Unlike the stages, they take both signs.  The decay must be real and positive.
        """
        binomials = np.asarray(
            [[math.comb(m, i) for i in range(self.order)] for m in range(self.order)]
        )
        powers = _powers(-2 * self.decay, self.order)[jnp.newaxis, :]
        return jnp.sqrt(2 * self.decay) * binomials * powers

    def laguerre(self, state: PyTree) -> PyTree:
        """Return the Laguerre components of the state, with the same structure as the state."""
        matrix = self.laguerre_matrix()
        pattern = _lower_triangle(self.order)
        return tree.map(lambda stages: mix_stages(matrix, stages, pattern), state)
