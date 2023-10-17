from __future__ import annotations

from functools import partial
from typing import Generic

import jax.numpy as jnp
from jax.tree_util import tree_map, tree_reduce
from typing_extensions import override

from ..annotations import JaxBooleanArray, JaxRealArray
from ..dataclasses import dataclass
from ..math_tools import divide_nonnegative
from .augmented import AugmentedState, State
from .iterated_function import Comparand, IteratedFunction, Parameters, Trajectory

__all__ = ['ComparingState', 'ComparingIteratedFunction']


@dataclass
class ComparingState(AugmentedState[State], Generic[State, Comparand]):
    last_state: Comparand


@dataclass
class ComparingIteratedFunction(
        IteratedFunction[Parameters, State, Comparand, Trajectory,
                         ComparingState[State, Comparand]],
        Generic[Parameters, State, Comparand, Trajectory]):
    """An iterated function that converges when part of two successive states are nearly equal."""

    # Implemented methods --------------------------------------------------------------------------
    @override
    def initial_augmented(self, initial_state: State) -> ComparingState[State, Comparand]:
        return ComparingState(current_state=initial_state,
                              iterations=0,
                              last_state=self.extract_comparand(initial_state))

    @override
    def expected_state(self, theta: Parameters, state: State) -> State:
        return self.sampled_state(theta, state)

    @override
    def iterate_augmented(self,
                          new_state: State,
                          augmented: ComparingState[State, Comparand]) -> (
                              ComparingState[State, Comparand]):
        return ComparingState(current_state=new_state,
                              iterations=augmented.iterations + 1,
                              last_state=self.extract_comparand(augmented.current_state))

    @override
    def converged(self, augmented: ComparingState[State, Comparand]) -> JaxBooleanArray:
        return tree_reduce(jnp.logical_and,
                           tree_map(partial(jnp.allclose, rtol=self.rtol, atol=self.atol),
                                    self.extract_comparand(augmented.current_state),
                                    augmented.last_state),
                           jnp.asarray(True))

    @override
    def minimum_tolerances(self, augmented: ComparingState[State, Comparand]
                           ) -> tuple[JaxRealArray, JaxRealArray]:
        """The minimum tolerances that would lead to convergence now.

        Returns:
            The minimum value of atol that would lead to convergence now.
            The minimum value of rtol that would lead to convergence now.
        """
        comparand = self.extract_comparand(augmented.current_state)
        abs_last = tree_map(jnp.abs, augmented.last_state)
        delta = tree_map(jnp.abs, tree_map(jnp.subtract, comparand, augmented.last_state))
        delta_over_b = tree_map(divide_nonnegative, delta, abs_last)
        minium_atol = tree_reduce(jnp.maximum,
                                  tree_map(jnp.amax, delta), jnp.asarray(0.0))
        minium_rtol = tree_reduce(jnp.maximum,
                                  tree_map(jnp.amax, delta_over_b), jnp.asarray(0.0))
        return minium_atol, minium_rtol
