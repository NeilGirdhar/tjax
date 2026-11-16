"""A buffer of the most recent values of a signal."""

from __future__ import annotations

from typing import override

import jax.numpy as jnp
from jax import tree

from tjax._src.annotations import JaxComplexArray, PyTree, RealNumeric
from tjax.dataclasses import dataclass, field

from .filter import Filter


@dataclass
class DelayLine(Filter[PyTree]):
    """The signal at each of the most recent lags.

    Each leaf's state and value have shape (length, *s), where entry L is the signal from L updates
    ago.  Unlike the other filters, its memory grows with the longest lag it holds, and it has no
    kernel to choose, so it shows a signal's history as it was.  Each update fills one slot, so the
    updates should be equally spaced; the time step is ignored.

    Args:
        length: The number of lags held.
    """

    length: int = field(static=True)

    @override
    def init(self, template: PyTree) -> PyTree:
        return tree.map(lambda leaf: jnp.zeros((self.length, *leaf.shape), leaf.dtype), template)

    @override
    def update(self, state: PyTree, drift: PyTree, time_step: RealNumeric = 1.0) -> PyTree:
        def push(buffer: JaxComplexArray, leaf: JaxComplexArray) -> JaxComplexArray:
            return jnp.concatenate([leaf[None].astype(buffer.dtype), buffer[:-1]])

        return tree.map(push, state, drift)

    @override
    def value(self, state: PyTree) -> PyTree:
        return state
