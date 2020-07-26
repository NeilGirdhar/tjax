from __future__ import annotations

from functools import partial
from typing import Generic, Tuple, TypeVar

from jax import numpy as jnp
from jax.tree_util import tree_map, tree_multimap, tree_reduce

from tjax import Generator, PyTree, Tensor, dataclass

from ..leaky_integral import leaky_integrate
from .augmented import AugmentedState, State
from .combinator import IteratedFunctionWithCombinator
from .iterated_function import IteratedFunction, Parameters

__all__ = ['StochasticState', 'StochasticIteratedFunction',
           'StochasticIteratedFunctionWithCombinator']


Comparand = TypeVar('Comparand', bound=PyTree)


@dataclass
class StochasticState(AugmentedState[State], Generic[State, Comparand]):

    mean_state: Comparand
    second_moment_state: Comparand
    rng: Generator


# https://github.com/python/mypy/issues/8539
@dataclass  # type: ignore
class StochasticIteratedFunction(
        IteratedFunction[Parameters, State, StochasticState[State, Comparand]],
        Generic[Parameters, State, Comparand]):

    initial_rng: Generator
    decay: float

    # Implemented methods --------------------------------------------------------------------------
    def initial_state(self, initial_state: State) -> StochasticState[State, Comparand]:
        mean_state, second_moment_state = self._sufficient_statistics(initial_state)
        return StochasticState(current_state=initial_state,
                               iterations=0,
                               mean_state=mean_state,
                               second_moment_state=second_moment_state,
                               rng=self.initial_rng)

    def iterate_augmented(self,
                          theta: Parameters,
                          augmented: StochasticState[State, Comparand]) -> (
                              StochasticState[State, Comparand]):
        def f(value: Tensor, drift: Tensor) -> Tensor:
            return leaky_integrate(value, 1.0, drift, self.decay, leaky_average=True)

        new_state, new_rng = self.stochastic_iterate_state(
            theta, augmented.current_state, augmented.rng)
        mean_state, second_moment_state = self._sufficient_statistics(augmented.current_state)
        new_mean_state = tree_multimap(f, augmented.mean_state, mean_state)
        new_second_moment_state = tree_multimap(f, augmented.second_moment_state,
                                                second_moment_state)
        return StochasticState(current_state=new_state,
                               iterations=augmented.iterations + 1,
                               mean_state=new_mean_state,
                               second_moment_state=new_second_moment_state,
                               rng=new_rng)

    def converged(self, augmented: StochasticState[State, Comparand]) -> Tensor:
        data_weight = leaky_integrate(0.0, augmented.iterations, 1.0, self.decay,
                                      leaky_average=True)
        mean_squared = tree_map(jnp.square, augmented.mean_state)
        return tree_reduce(jnp.logical_and,
                           tree_multimap(partial(jnp.allclose,
                                                 rtol=self.rtol * data_weight,
                                                 atol=self.atol * data_weight),
                                         augmented.second_moment_state,
                                         mean_squared))

    # Abstract methods -----------------------------------------------------------------------------
    def extract_comparand(self, state: State) -> Comparand:
        """
        Returns: A pytree that will be compared in successive states to check whether the state has
            converged.
        """
        raise NotImplementedError

    def stochastic_iterate_state(self,
                                 theta: Parameters,
                                 state: State,
                                 rng: Generator) -> Tuple[State, Generator]:
        raise NotImplementedError

    # New methods ----------------------------------------------------------------------------------
    def convergence_atol(self, augmented: StochasticState[State, Comparand]) -> Tensor:
        data_weight = leaky_integrate(0.0, augmented.iterations, 1.0, self.decay,
                                      leaky_average=True)
        mean_squared = tree_map(jnp.square, augmented.mean_state)
        variance = tree_multimap(jnp.subtract, augmented.second_moment_state, mean_squared)
        return tree_reduce(jnp.maximum, tree_map(jnp.amax, variance)) / data_weight

    # Private methods ------------------------------------------------------------------------------
    def _sufficient_statistics(self, state: State) -> Tuple[Comparand, Comparand]:
        comparand = self.extract_comparand(state)
        squared_comparand = tree_map(jnp.square, comparand)
        return comparand, squared_comparand


class StochasticIteratedFunctionWithCombinator(
        IteratedFunctionWithCombinator[Parameters, State, StochasticState[State, Comparand]],
        StochasticIteratedFunction[Parameters, State, Comparand],
        Generic[Parameters, State, Comparand]):
    pass
