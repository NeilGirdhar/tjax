"""Leaky averages that start at zero without underestimating the signal."""

from __future__ import annotations

from typing import override

import jax.numpy as jnp
from jax import tree

from tjax._src.annotations import JaxRealArray, PyTree, RealNumeric
from tjax._src.math_tools import divide_or_zero, sesquilinear_outer
from tjax.dataclasses import dataclass

from .filter import Filter, state_dtype
from .leaky import leaky_integrate


@dataclass
class LeakyAverageState[T: PyTree]:
    """The state of a `LeakyAverage`.

    Args:
        total: The leaky average of the signal, which starts at zero and has the signal's structure.
        data_weight: The leaky average of one, which is the fraction of the total that holds data.
    """

    total: T
    data_weight: JaxRealArray


@dataclass
class LeakyAverage(Filter[LeakyAverageState]):
    """The leaky average of a signal, corrected for its zero initialization.

    A leaky average that starts at zero underestimates the signal until it has seen enough data.
    This filter divides by the weight of the data seen so far, as Adam does, so its value is an
    average of the signal from the first step.  The kernel is proportional to exp(-decay u).  The
    state's total and the value keep the template's pytree structure and type.

    Args:
        decay: The decay rate.  It must be positive, since an average with no decay admits no data.
    """

    decay: JaxRealArray

    @override
    def init[T: PyTree](self, template: T) -> LeakyAverageState[T]:
        total = tree.map(lambda leaf: jnp.zeros_like(leaf, dtype=state_dtype(leaf, self)), template)
        return LeakyAverageState(total, jnp.zeros_like(self.decay))

    @override
    def update[T: PyTree](
        self, state: LeakyAverageState[T], drift: T, time_step: RealNumeric = 1.0
    ) -> LeakyAverageState[T]:
        total: T = tree.map(
            lambda value, leaf: leaky_integrate(
                value, time_step, leaf, self.decay, leaky_average=True
            ),
            state.total,
            drift,
        )
        data_weight = jnp.asarray(
            leaky_integrate(
                state.data_weight,
                time_step,
                jnp.ones_like(state.data_weight),
                self.decay,
                leaky_average=True,
            )
        )
        return LeakyAverageState(total, data_weight)

    @override
    def value[T: PyTree](self, state: LeakyAverageState[T]) -> T:
        return tree.map(lambda total: divide_or_zero(total, state.data_weight), state.total)


@dataclass
class LeakyMoments(Filter[LeakyAverageState]):
    """The leaky mean and second moment of a vector signal.

    Like `LeakyAverage`, it divides by the weight of the data seen so far, so it doesn't
    underestimate early on.

    Each leaf of the signal has shape (..., n), and its value is a pair of the mean, with shape
    (..., n), and the second moment E[x conj(x)^T], with shape (..., n, n).  To track the cross
    moments of two signals, concatenate them into one.

    Args:
        decay: The decay rate.  It must be positive.
    """

    decay: JaxRealArray

    @override
    def init(self, template: PyTree) -> LeakyAverageState:
        return self._average().init(self._moments(template))

    @override
    def update(
        self, state: LeakyAverageState, drift: PyTree, time_step: RealNumeric = 1.0
    ) -> LeakyAverageState:
        return self._average().update(state, self._moments(drift), time_step)

    @override
    def value(self, state: LeakyAverageState) -> PyTree:
        return self._average().value(state)

    def _average(self) -> LeakyAverage:
        return LeakyAverage(self.decay)

    @staticmethod
    def _moments(signal: PyTree) -> PyTree:
        return tree.map(lambda leaf: (leaf, sesquilinear_outer(leaf, leaf)), signal)
