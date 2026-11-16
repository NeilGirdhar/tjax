"""Parallel leaky integrals at different decay rates."""

from __future__ import annotations

from typing import override

import jax.numpy as jnp
import numpy as np

from tjax._src.annotations import JaxComplexArray, RealNumeric
from tjax.dataclasses import dataclass

from .filter import StagedFilter
from .leaky import step_weight


@dataclass
class ExponentialBank(StagedFilter):
    """Leaky integrals of the same signal at different decay rates, read out through weights.

    Each leaf's state has shape (order, *s), where order is the number of decay rates.  The kernel
    is psi(u) = sum_j weights_j exp(-decays_j u).  Every term peaks at no delay, so the bank cannot
    form a delayed peak, but mixing slow and fast rates gives long or heavy tails.  The kernel is
    nonnegative when the weights are nonnegative.

    Args:
        decays: The decay rates, with shape (order,).  Their real components must be nonnegative.
        weights: The weights of the integrals, with shape (order,).
    """

    decays: JaxComplexArray
    weights: JaxComplexArray

    @property
    @override
    def order(self) -> int:
        return self.decays.shape[0]

    @property
    @override
    def readout(self) -> JaxComplexArray:
        return self.weights

    @property
    @override
    def transition_pattern(self) -> np.ndarray:
        """The diagonal, since the integrals are independent.

        This override is only for speed: each step skips the zeros off the diagonal.
        """
        return np.eye(self.order, dtype=bool)

    @override
    def discretize(self, time_step: RealNumeric) -> tuple[JaxComplexArray, JaxComplexArray]:
        x = self.decays * time_step
        return jnp.diag(jnp.exp(-x)), jnp.asarray(time_step * step_weight(x))
