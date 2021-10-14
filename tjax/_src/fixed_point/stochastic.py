from __future__ import annotations

from functools import partial
from typing import Generic, Tuple

import jax.numpy as jnp
from jax.tree_util import tree_map, tree_reduce

from ..annotations import BooleanNumeric, ComplexNumeric, RealNumeric
from ..dataclasses import dataclass
from ..leaky_integral import leaky_data_weight, leaky_integrate
from ..tools import abs_square, divide_nonnegative
from .augmented import AugmentedState, State
from .combinator import Differentiand, IteratedFunctionWithCombinator
from .iterated_function import Comparand, IteratedFunction, Parameters, Trajectory

__all__ = ['StochasticState', 'StochasticIteratedFunction',
           'StochasticIteratedFunctionWithCombinator']


@dataclass
class StochasticState(AugmentedState[State], Generic[State, Comparand]):
    mean_state: Comparand
    second_moment_state: Comparand


@dataclass
class StochasticIteratedFunction(
        IteratedFunction[Parameters, State, Comparand, Trajectory,
                         StochasticState[State, Comparand]],
        Generic[Parameters, State, Comparand, Trajectory]):
    convergence_detection_decay: RealNumeric

    # Implemented methods --------------------------------------------------------------------------
    def initial_augmented(self, initial_state: State) -> StochasticState[State, Comparand]:
        comparand = self.extract_comparand(initial_state)
        zero_comparand = tree_map(jnp.zeros_like, comparand)
        return StochasticState(current_state=initial_state,
                               iterations=0,
                               mean_state=zero_comparand,
                               second_moment_state=zero_comparand)

    def iterate_augmented(self,
                          new_state: State,
                          augmented: StochasticState[State, Comparand]) -> (
                              StochasticState[State, Comparand]):
        def f(value: ComplexNumeric, drift: ComplexNumeric) -> ComplexNumeric:
            return leaky_integrate(value, 1.0, drift, self.convergence_detection_decay,
                                   leaky_average=True)

        mean_state, second_moment_state = self._sufficient_statistics(augmented.current_state)
        new_mean_state = tree_map(f, augmented.mean_state, mean_state)
        new_second_moment_state = tree_map(f, augmented.second_moment_state, second_moment_state)
        return StochasticState(current_state=new_state,
                               iterations=augmented.iterations + 1,
                               mean_state=new_mean_state,
                               second_moment_state=new_second_moment_state)

    def converged(self, augmented: StochasticState[State, Comparand]) -> BooleanNumeric:
        data_weight = leaky_data_weight(augmented.iterations, self.convergence_detection_decay)
        mean_squared = tree_map(abs_square, augmented.mean_state)
        return tree_reduce(jnp.logical_and,
                           tree_map(partial(jnp.allclose, rtol=self.rtol * data_weight,
                                            atol=self.atol * data_weight),
                                    augmented.second_moment_state,
                                    mean_squared),
                           True)

    def minimum_tolerances(self,
                           augmented: StochasticState[State, Comparand]) -> Tuple[RealNumeric,
                                                                                  RealNumeric]:
        """
        Returns:
            The minimum value of atol that would lead to convergence now.
            The minimum value of rtol that would lead to convergence now.
        """
        data_weight = leaky_data_weight(augmented.iterations, self.convergence_detection_decay)
        mean_squared = tree_map(abs_square, augmented.mean_state)
        variance = tree_map(jnp.subtract, augmented.second_moment_state, mean_squared)
        scaled_variance = tree_map(divide_nonnegative, variance, mean_squared)

        minimum_atol = divide_nonnegative(tree_reduce(jnp.maximum, tree_map(jnp.amax, variance),
                                                      0.0),
                                          data_weight)
        minimum_rtol = divide_nonnegative(tree_reduce(jnp.maximum,
                                                      tree_map(jnp.amax, scaled_variance), 0.0),
                                          data_weight)
        assert not isinstance(minimum_atol, complex)
        assert not isinstance(minimum_rtol, complex)
        return minimum_atol, minimum_rtol

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
