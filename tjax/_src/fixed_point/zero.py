from __future__ import annotations

from typing import Any, Generic

import jax.numpy as jnp
from jax.tree_util import tree_map, tree_reduce
from typing_extensions import override

from ..annotations import JaxBooleanArray, JaxRealArray
from ..dataclasses import dataclass
from .augmented import AugmentedState, State
from .combinator import Differentiand, IteratedFunctionWithCombinator
from .iterated_function import Comparand, IteratedFunction, Parameters, Trajectory

__all__ = ['ZeroIteratedFunction', 'ZeroIteratedFunctionWithCombinator']


@dataclass
class ZeroIteratedFunction(
        IteratedFunction[Parameters, State, Comparand, Trajectory,
                         AugmentedState[State]],
        Generic[Parameters, State, Comparand, Trajectory]):
    """An iterated function that converges when a part of the state is zero."""

    # Implemented methods --------------------------------------------------------------------------
    @override
    def initial_augmented(self, initial_state: State) -> AugmentedState[State]:
        return AugmentedState(current_state=initial_state,
                              iterations=0)

    @override
    def expected_state(self, theta: Parameters, state: State) -> State:
        return self.sampled_state(theta, state)

    @override
    def iterate_augmented(self,
                          new_state: State,
                          augmented: AugmentedState[State]
                          ) -> AugmentedState[State]:
        return AugmentedState(current_state=new_state,
                              iterations=augmented.iterations + 1)

    @override
    def converged(self, augmented: AugmentedState[State]) -> JaxBooleanArray:
        comparand = self.extract_comparand(augmented.current_state)

        def all_close_zero(x: Any) -> JaxBooleanArray:
            return jnp.all(jnp.abs(x) < self.atol)

        return tree_reduce(jnp.logical_and,
                           tree_map(all_close_zero, comparand),
                           jnp.asarray(True))

    @override
    def minimum_tolerances(self, augmented: AugmentedState[State]
                           ) -> tuple[JaxRealArray, JaxRealArray]:
        """The minimum tolerances that would lead to convergence now.

        Returns:
            The minimum value of atol that would lead to convergence now.
            The minimum value of rtol that would lead to convergence now.
        """
        comparand = self.extract_comparand(augmented.current_state)
        abs_comparand = tree_map(jnp.abs, comparand)
        minium_atol = tree_reduce(jnp.maximum,
                                  tree_map(jnp.amax, abs_comparand), jnp.asarray(0.0))
        return minium_atol, jnp.array(0.0)


class ZeroIteratedFunctionWithCombinator(
        IteratedFunctionWithCombinator[Parameters, State, Comparand, Differentiand, Trajectory,
                                       AugmentedState[State]],
        ZeroIteratedFunction[Parameters, State, Comparand, Trajectory],
        Generic[Parameters, State, Comparand, Differentiand, Trajectory]):
    pass
