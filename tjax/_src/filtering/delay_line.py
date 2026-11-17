"""A finite window over the most recent values of a signal."""

from __future__ import annotations

from typing import override

import jax.numpy as jnp
import numpy as np

from tjax._src.annotations import JaxComplexArray, RealNumeric
from tjax.dataclasses import dataclass

from .filter import StagedFilter


@dataclass
class DelayLine(StagedFilter):
    """The most recent values of a signal, read out through a weight for each lag.

    Each leaf's state has shape (order, *s), where order is the length of the weights and stage L
    is the signal from L updates ago.  The weights are the lag weights themselves, so the kernel can
    be any finite window, including a hard cutoff or a pure delay, which the continuous-time filters
    cannot form.  Its memory grows with the longest lag.

    Each update fills one slot, so the updates should be equally spaced; the time step is ignored.

    Args:
        weights: The weight of each lag, with shape (order,), starting with the signal from the
            latest update.
    """

    weights: JaxComplexArray

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
        """The subdiagonal, since each stage takes the value of the one before it.

        This override is only for speed: each step skips every other entry.
        """
        return np.eye(self.order, k=-1, dtype=bool)

    @override
    def discretize(self, time_step: RealNumeric) -> tuple[JaxComplexArray, JaxComplexArray]:
        """Return the shift that moves each value one lag older, ignoring the time step."""
        dtype = self.weights.dtype
        return jnp.eye(self.order, k=-1, dtype=dtype), jnp.eye(self.order, dtype=dtype)[0]
