from __future__ import annotations

from functools import partial
from typing import Generic

import jax.numpy as jnp
from jax import Array, tree
from typing_extensions import override

from tjax.dataclasses import dataclass

from ..annotations import ComplexArray, JaxBooleanArray, JaxRealArray
from ..leaky_integral import leaky_data_weight, leaky_integrate
from ..math_tools import abs_square, divide_nonnegative
from .augmented import AugmentedState, State
from .combinator import Differentiand, IteratedFunctionWithCombinator
from .iterated_function import Comparand, IteratedFunction, Parameters, Trajectory


@dataclass
class StochasticState(AugmentedState[State], Generic[State, Comparand]):
    mean_state: Comparand
    second_moment_state: Comparand


@dataclass
class StochasticIteratedFunction(
        IteratedFunction[Parameters, State, Comparand, Trajectory,
                         StochasticState[State, Comparand]],
        Generic[Parameters, State, Comparand, Trajectory]):
    """An iterated function that generates a stochastic next state.

    It converges when an exponentially decaying window has nearly zero variance.
    """

    convergence_detection_decay: JaxRealArray

    # Implemented methods --------------------------------------------------------------------------
    @override
    def initial_augmented(self, initial_state: State) -> StochasticState[State, Comparand]:
        comparand = self.extract_comparand(initial_state)
        zero_comparand = tree.map(jnp.zeros_like, comparand)
        return StochasticState(current_state=initial_state,
                               iterations=jnp.asarray(0),
                               mean_state=zero_comparand,
                               second_moment_state=zero_comparand)

    @override
    def iterate_augmented(self,
                          new_state: State,
                          augmented: StochasticState[State, Comparand]) -> (
                              StochasticState[State, Comparand]):
        def f(value: ComplexArray, drift: ComplexArray) -> ComplexArray:
            return leaky_integrate(value, jnp.asarray(1.0), drift, self.convergence_detection_decay,
                                   leaky_average=True)

        mean_state, second_moment_state = self._sufficient_statistics(augmented.current_state)
        new_mean_state = tree.map(f, augmented.mean_state, mean_state)
        new_second_moment_state = tree.map(f, augmented.second_moment_state, second_moment_state)
        return StochasticState(current_state=new_state,
                               iterations=augmented.iterations + 1,
                               mean_state=new_mean_state,
                               second_moment_state=new_second_moment_state)

    @override
    def converged(self, augmented: StochasticState[State, Comparand]) -> JaxBooleanArray:
        data_weight = leaky_data_weight(augmented.iterations, self.convergence_detection_decay)
        mean_squared = tree.map(abs_square, augmented.mean_state)
        return tree.reduce(jnp.logical_and,
                           tree.map(partial(jnp.allclose, rtol=self.rtol * data_weight,
                                            atol=self.atol * data_weight),
                                    augmented.second_moment_state,
                                    mean_squared),
                           jnp.asarray(True))

    @override
    def minimum_tolerances(self,
                           augmented: StochasticState[State, Comparand]
                           ) -> tuple[JaxRealArray, JaxRealArray]:
        """The minimum tolerances that would lead to convergence now.

        Returns:
            The minimum value of atol that would lead to convergence now.
            The minimum value of rtol that would lead to convergence now.
        """
        data_weight = leaky_data_weight(augmented.iterations, self.convergence_detection_decay)
        mean_squared = tree.map(abs_square, augmented.mean_state)
        variance = tree.map(jnp.subtract, augmented.second_moment_state, mean_squared)
        scaled_variance = tree.map(divide_nonnegative, variance, mean_squared)

        minimum_atol = divide_nonnegative(tree.reduce(jnp.maximum,
                                                      tree.map(jnp.amax, variance),
                                                      jnp.asarray(0.0)),
                                          data_weight)
        minimum_rtol = divide_nonnegative(tree.reduce(jnp.maximum,
                                                      tree.map(jnp.amax, scaled_variance),
                                                      jnp.asarray(0.0)),
                                          data_weight)
        assert isinstance(minimum_atol, Array)
        assert isinstance(minimum_rtol, Array)
        return minimum_atol, minimum_rtol

    # Private methods ------------------------------------------------------------------------------
    def _sufficient_statistics(self, state: State) -> tuple[Comparand, Comparand]:
        comparand = self.extract_comparand(state)
        squared_comparand = tree.map(abs_square, comparand)
        return comparand, squared_comparand


class StochasticIteratedFunctionWithCombinator(
        IteratedFunctionWithCombinator[Parameters, State, Comparand, Differentiand, Trajectory,
                                       StochasticState[State, Comparand]],
        StochasticIteratedFunction[Parameters, State, Comparand, Trajectory],
        Generic[Parameters, State, Comparand, Differentiand, Trajectory]):
    pass
