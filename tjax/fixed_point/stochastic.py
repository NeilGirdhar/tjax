from __future__ import annotations

from functools import partial
from typing import Generic, Tuple

import jax.numpy as jnp
from chex import Array
from jax.tree_util import tree_map, tree_multimap, tree_reduce

from ..dataclass import dataclass
from ..leaky_integral import leaky_data_weight, leaky_integrate
from ..tools import abs_square, safe_divide
from .augmented import AugmentedState, State
from .combinator import Differentiand, IteratedFunctionWithCombinator
from .iterated_function import Comparand, IteratedFunction, Parameters, Trajectory

__all__ = ['StochasticState', 'StochasticIteratedFunction',
           'StochasticIteratedFunctionWithCombinator']


@dataclass
class StochasticState(AugmentedState[State], Generic[State, Comparand]):

    mean_state: Comparand
    second_moment_state: Comparand


# https://github.com/python/mypy/issues/8539
@dataclass  # type: ignore
class StochasticIteratedFunction(
        IteratedFunction[Parameters, State, Comparand, Trajectory,
                         StochasticState[State, Comparand]],
        Generic[Parameters, State, Comparand, Trajectory]):

    convergence_detection_decay: float

    # Implemented methods --------------------------------------------------------------------------
    def initial_augmented(self, initial_state: State) -> StochasticState[State, Comparand]:
        mean_state, second_moment_state = self._sufficient_statistics(initial_state)
        return StochasticState(current_state=initial_state,
                               iterations=0,
                               mean_state=mean_state,
                               second_moment_state=second_moment_state)

    def iterate_augmented(self,
                          new_state: State,
                          augmented: StochasticState[State, Comparand]) -> (
                              StochasticState[State, Comparand]):
        def f(value: Array, drift: Array) -> Array:
            return leaky_integrate(value, 1.0, drift, self.convergence_detection_decay,
                                   leaky_average=True)

        mean_state, second_moment_state = self._sufficient_statistics(augmented.current_state)
        new_mean_state = tree_multimap(f, augmented.mean_state, mean_state)
        new_second_moment_state = tree_multimap(f, augmented.second_moment_state,
                                                second_moment_state)
        return StochasticState(current_state=new_state,
                               iterations=augmented.iterations + 1,
                               mean_state=new_mean_state,
                               second_moment_state=new_second_moment_state)

    def converged(self, augmented: StochasticState[State, Comparand]) -> Array:
        data_weight = leaky_data_weight(augmented.iterations, self.convergence_detection_decay)
        mean_squared = tree_map(abs_square, augmented.mean_state)
        return tree_reduce(jnp.logical_and,
                           tree_multimap(partial(jnp.allclose,
                                                 rtol=self.rtol * data_weight,
                                                 atol=self.atol * data_weight),
                                         augmented.second_moment_state,
                                         mean_squared),
                           True)

    def minimum_tolerances(self, augmented: StochasticState[State, Comparand]) -> Tuple[Array,
                                                                                        Array]:
        """
        Returns:
            The minimum value of atol that would lead to convergence now.
            The minimum value of rtol that would lead to convergence now.
        """
        data_weight = leaky_data_weight(augmented.iterations, self.convergence_detection_decay)
        mean_squared = tree_map(abs_square, augmented.mean_state)
        variance = tree_multimap(jnp.subtract, augmented.second_moment_state, mean_squared)
        scaled_variance = tree_multimap(safe_divide, variance, mean_squared)

        minium_atol = safe_divide(tree_reduce(jnp.maximum, tree_map(jnp.amax, variance)),
                                  data_weight)
        minium_rtol = safe_divide(tree_reduce(jnp.maximum, tree_map(jnp.amax, scaled_variance)),
                                  data_weight)
        return minium_atol, minium_rtol

    # Private methods ------------------------------------------------------------------------------
    def _sufficient_statistics(self, state: State) -> Tuple[Comparand, Comparand]:
        comparand = self.extract_comparand(state)
        squared_comparand = tree_map(abs_square, comparand)
        return comparand, squared_comparand


class StochasticIteratedFunctionWithCombinator(
        IteratedFunctionWithCombinator[Parameters, State, Comparand, Differentiand, Trajectory,
                                       StochasticState[State, Comparand]],
        StochasticIteratedFunction[Parameters, State, Comparand, Trajectory],
        Generic[Parameters, State, Comparand, Differentiand, Trajectory]):
    pass
