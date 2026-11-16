"""The exact step of a general linear system, and the filter built on it."""

from __future__ import annotations

from typing import override

import jax.numpy as jnp
from jax.scipy.linalg import expm

from tjax._src.annotations import JaxComplexArray, RealNumeric
from tjax.dataclasses import dataclass

from .filter import StagedFilter


@dataclass
class LinearFilter(StagedFilter):
    """A filter whose stages follow ds/dt = system @ s + input_gain * c.

    This is the general case.  Each step takes a matrix exponential of size order + 1, so prefer
    the specialized filters when they apply.  The kernel is nonnegative when the system matrix has
    nonnegative off-diagonal entries and the input and output gains are nonnegative.

    Args:
        system: The system matrix, with shape (order, order).
        input_gain: The input vector, with shape (order,).
        output_gain: The weights that combine the stages into the value, with shape (order,).
    """

    system: JaxComplexArray
    input_gain: JaxComplexArray
    output_gain: JaxComplexArray

    @property
    @override
    def order(self) -> int:
        return self.system.shape[-1]

    @property
    @override
    def readout(self) -> JaxComplexArray:
        return self.output_gain

    @override
    def discretize(self, time_step: RealNumeric) -> tuple[JaxComplexArray, JaxComplexArray]:
        """Return the exact step of the system with the signal held over the step.

        The transition is exp(system * time_step), and the input weights are the integral of
        exp(system * u) @ input_gain over u from 0 to time_step.  One matrix exponential of the
        augmented matrix [[system, input_gain], [0, 0]] gives both (Van Loan's method), so the
        result stays finite when the system matrix is singular.
        """
        n = self.order
        dtype = jnp.result_type(self.system, self.input_gain, time_step)
        augmented = jnp.zeros((n + 1, n + 1), dtype=dtype)
        augmented = augmented.at[:n, :n].set(self.system).at[:n, n].set(self.input_gain)
        exponential = expm(augmented * time_step)
        return exponential[:n, :n], exponential[:n, n]
