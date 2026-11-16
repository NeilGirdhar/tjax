"""The leaky integral as a filter."""

from __future__ import annotations

from typing import override

import jax.numpy as jnp
from jax import tree

from tjax._src.annotations import JaxComplexArray, PyTree, RealNumeric
from tjax.dataclasses import dataclass

from .filter import Filter, state_dtype
from .leaky import leaky_integrate, step_weight


@dataclass
class LeakyIntegral(Filter[PyTree]):
    """The leaky integral of a signal, whose state has the signal's shape.

    The kernel is exp(-decay u).  This equals an `ErlangChain` with one stage and a weight of one,
    except that the state has no stage axis.

    Args:
        decay: The decay rate.  Its real component must be nonnegative.
    """

    decay: JaxComplexArray

    @override
    def init(self, template: PyTree) -> PyTree:
        return tree.map(lambda leaf: jnp.zeros_like(leaf, dtype=state_dtype(leaf, self)), template)

    @override
    def update(self, state: PyTree, drift: PyTree, time_step: RealNumeric = 1.0) -> PyTree:
        return tree.map(
            lambda value, leaf: leaky_integrate(value, time_step, leaf, self.decay), state, drift
        )

    @override
    def value(self, state: PyTree) -> PyTree:
        return state

    def lag_weights(self, num_lags: int, time_step: RealNumeric = 1.0) -> JaxComplexArray:
        """Return the value's weight on the signal at each lag, with shape (num_lags,).

        Lag L is the signal held over the step that ended L steps before the value is read.
        """
        x = self.decay * time_step
        return jnp.asarray(time_step * step_weight(x) * jnp.exp(-x * jnp.arange(num_lags)))
