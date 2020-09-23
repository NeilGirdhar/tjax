from __future__ import annotations

from functools import partial
from typing import Generic

from chex import Array
from jax import numpy as jnp
from jax.tree_util import tree_multimap, tree_reduce

from tjax import dataclass

from .augmented import AugmentedState, State
from .iterated_function import Comparand, IteratedFunction, Parameters, Trajectory

__all__ = ['ComparingState', 'ComparingIteratedFunction']


@dataclass
class ComparingState(AugmentedState[State], Generic[State, Comparand]):

    last_state: Comparand


# https://github.com/python/mypy/issues/8539
@dataclass  # type: ignore
class ComparingIteratedFunction(
        IteratedFunction[Parameters, State, Comparand, Trajectory,
                         ComparingState[State, Comparand]],
        Generic[Parameters, State, Comparand, Trajectory]):

    # Implemented methods --------------------------------------------------------------------------
    def initial_augmented(self, initial_state: State) -> ComparingState[State, Comparand]:
        return ComparingState(current_state=initial_state,
                              iterations=0,
                              last_state=self.extract_comparand(initial_state))

    def expected_state(self, theta: Parameters, state: State) -> State:
        return self.sampled_state(theta, state)

    def iterate_augmented(self,
                          new_state: State,
                          augmented: ComparingState[State, Comparand]) -> (
                              ComparingState[State, Comparand]):
        return ComparingState(current_state=new_state,
                              iterations=augmented.iterations + 1,
                              last_state=self.extract_comparand(augmented.current_state))

    def converged(self, augmented: ComparingState[State, Comparand]) -> Array:
        return tree_reduce(jnp.logical_and,
                           tree_multimap(partial(jnp.allclose, rtol=self.rtol, atol=self.atol),
                                         self.extract_comparand(augmented.current_state),
                                         augmented.last_state))
