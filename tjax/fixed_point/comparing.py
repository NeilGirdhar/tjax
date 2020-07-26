from __future__ import annotations

from functools import partial
from typing import Generic, TypeVar

from jax import numpy as jnp
from jax.tree_util import tree_multimap, tree_reduce

from tjax import PyTree, Tensor, dataclass

from .augmented import AugmentedState, State
from .iterated_function import IteratedFunction, Parameters

__all__ = ['ComparingState', 'ComparingIteratedFunction']


Comparand = TypeVar('Comparand', bound=PyTree)


@dataclass
class ComparingState(AugmentedState[State], Generic[State, Comparand]):

    last_state: Comparand


# https://github.com/python/mypy/issues/8539
@dataclass  # type: ignore
class ComparingIteratedFunction(
        IteratedFunction[Parameters, State, ComparingState[State, Comparand]],
        Generic[Parameters, State, Comparand]):

    # Implemented methods --------------------------------------------------------------------------
    def initial_state(self, initial_state: State) -> ComparingState[State, Comparand]:
        return ComparingState(current_state=initial_state,
                              iterations=0,
                              last_state=self.extract_comparand(initial_state))

    def iterate_augmented(self,
                          theta: Parameters,
                          augmented: ComparingState[State, Comparand]) -> (
                              ComparingState[State, Comparand]):
        return ComparingState(current_state=self.iterate_state(theta, augmented.current_state),
                              iterations=augmented.iterations + 1,
                              last_state=self.extract_comparand(augmented.current_state))

    def converged(self, augmented: ComparingState[State, Comparand]) -> Tensor:
        return tree_reduce(jnp.logical_and,
                           tree_multimap(partial(jnp.allclose, rtol=self.rtol, atol=self.atol),
                                         self.extract_comparand(augmented.current_state),
                                         augmented.last_state))

    # New methods ----------------------------------------------------------------------------------
    def extract_comparand(self, state: State) -> Comparand:
        """
        Returns: A pytree that will be compared in successive states to check whether the state has
            converged.
        """
        raise NotImplementedError
