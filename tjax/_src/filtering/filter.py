"""Base classes for filters that summarize the history of a signal in a fixed-size state."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import override

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree

from tjax._src.annotations import JaxArray, JaxComplexArray, PyTree, RealNumeric
from tjax.dataclasses import dataclass


@dataclass
class Filter[State: PyTree](ABC):
    """An online filter of a pytree signal whose state has a fixed size.

    The fields are the filter's parameters, and the state is a separate pytree, so one filter can
    advance many states.  Each update receives the signal as a rate held constant over the step, so
    a step of two seconds equals two steps of one second.
    """

    @abstractmethod
    def init(self, template: PyTree) -> State:
        """Return the initial state for signals shaped like the template."""
        raise NotImplementedError

    @abstractmethod
    def update(self, state: State, drift: PyTree, time_step: RealNumeric = 1.0) -> State:
        """Return the state after a step over which the signal equals the drift."""
        raise NotImplementedError

    @abstractmethod
    def value(self, state: State) -> PyTree:
        """Return the filtered signal, which has the same structure as the template."""
        raise NotImplementedError


def state_dtype(leaf: JaxArray, parameters: PyTree) -> jnp.dtype:
    """Return a dtype that holds both a signal leaf and anything the parameters produce."""
    return jnp.result_type(leaf, *tree.leaves(parameters))


def expand_stages(vector: JaxComplexArray, ndim: int) -> JaxComplexArray:
    """Reshape a vector over stages so that it broadcasts against stages of a leaf of rank ndim."""
    return jnp.reshape(vector, vector.shape + (1,) * ndim)


def mix_stages(
    matrix: JaxComplexArray, stages: JaxComplexArray, pattern: np.ndarray | None = None
) -> JaxComplexArray:
    """Return the matrix applied along the stage axis of one leaf's stages.

    The product is written out as one multiply-add per matrix entry for speed.  XLA fuses these into
    a single elementwise pass, whereas a tensordot over the short stage axis compiles to a matrix
    multiply and a second pass, which is several times slower on CPU.  Entries outside the pattern
    are skipped.

    Args:
        matrix: The matrix, with shape (order, order).
        stages: The stages, with shape (order, *s).
        pattern: A static boolean array, with shape (order, order), that marks the entries of the
            matrix that can be nonzero.  None marks every entry.
    """
    order = stages.shape[0]
    if pattern is None:
        pattern = np.ones((order, order), dtype=bool)
    rows = []
    for j in range(order):
        row = jnp.zeros_like(stages[0])
        for i in range(order):
            if pattern[j, i]:
                row += matrix[j, i] * stages[i]
        rows.append(row)
    return jnp.stack(rows)


def advance_stages(
    stages: JaxComplexArray,
    drift: JaxComplexArray,
    transition: JaxComplexArray,
    input_weight: JaxComplexArray,
    pattern: np.ndarray | None = None,
) -> JaxComplexArray:
    """Apply a discretized linear step to the stages of one leaf.

    Args:
        stages: The stages, with shape (order, *s).
        drift: The signal over the step, with shape s.
        transition: The matrix that carries the stages over the step, with shape (order, order).
        input_weight: The stages' response to a unit signal over the step, with shape (order,).
        pattern: The entries of the transition matrix that can be nonzero.  See `mix_stages`.
    """
    weighted_drift = expand_stages(input_weight, drift.ndim) * drift[jnp.newaxis]
    return mix_stages(transition, stages, pattern) + weighted_drift


@dataclass
class StagedFilter(Filter[PyTree], ABC):
    """A linear filter that holds a fixed number of stages for each element of the signal.

    Each leaf of shape s has state of shape (order, *s).  The stages evolve as
    ds/dt = A s + b c for some system matrix A and input vector b, and the value is a weighted sum
    of the stages.  Subclasses give the exact step of that system for a signal held constant over
    the step.

    The state and the value keep the template's pytree structure, so `init`, `update`, and `value`
    return the type of the pytree they are given.
    """

    @property
    @abstractmethod
    def order(self) -> int:
        """The number of stages per element."""
        raise NotImplementedError

    @property
    @abstractmethod
    def readout(self) -> JaxComplexArray:
        """The weights that combine the stages into the value, with shape (order,)."""
        raise NotImplementedError

    @property
    def transition_pattern(self) -> np.ndarray:
        """The entries of the transition matrix that can be nonzero, as a static boolean array.

        By default, every entry.  Subclasses with structured transitions narrow it for speed, so
        that each step skips the zeros.  It must not depend on traced values.
        """
        return np.ones((self.order, self.order), dtype=bool)

    @override
    def init[T: PyTree](self, template: T) -> T:
        return tree.map(
            lambda leaf: jnp.zeros((self.order, *leaf.shape), dtype=state_dtype(leaf, self)),
            template,
        )

    @override
    def update[T: PyTree](self, state: T, drift: T, time_step: RealNumeric = 1.0) -> T:
        transition, input_weight = self.discretize(time_step)
        return tree.map(
            lambda stages, leaf: advance_stages(
                stages, leaf, transition, input_weight, self.transition_pattern
            ),
            state,
            drift,
        )

    @override
    def value[T: PyTree](self, state: T) -> T:
        return tree.map(lambda stages: jnp.tensordot(self.readout, stages, axes=1), state)

    @abstractmethod
    def discretize(self, time_step: RealNumeric) -> tuple[JaxComplexArray, JaxComplexArray]:
        """Return the transition matrix and input weights of one step.

        The transition matrix has shape (order, order) and the input weights have shape (order,), so
        that stages' = transition @ stages + input_weights * drift.
        """
        raise NotImplementedError

    def basis_lag_weights(self, num_lags: int, time_step: RealNumeric = 1.0) -> JaxComplexArray:
        """Return each stage's weight on the signal at each lag, with shape (order, num_lags).

        Lag L is the signal held over the step that ended L steps before the stages are read.  These
        are the stages' responses to a unit signal over one step, followed by no signal.
        """
        transition, input_weight = self.discretize(time_step)

        def step(stages: JaxComplexArray, _: None) -> tuple[JaxComplexArray, JaxComplexArray]:
            return transition @ stages, stages

        _, responses = jax.lax.scan(step, input_weight, length=num_lags)
        return responses.T

    def lag_weights(self, num_lags: int, time_step: RealNumeric = 1.0) -> JaxComplexArray:
        """Return the value's weight on the signal at each lag, with shape (num_lags,).

        This is the filter's kernel integrated over each step.  See `basis_lag_weights`.
        """
        return self.readout @ self.basis_lag_weights(num_lags, time_step)
