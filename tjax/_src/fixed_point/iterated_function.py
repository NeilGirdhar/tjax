from __future__ import annotations

from functools import partial
from typing import Any, Generic, TypeVar

import jax.numpy as jnp
from jax import Array
from jax.lax import while_loop

from ..annotations import BooleanNumeric, IntegralNumeric, PyTree, RealNumeric
from ..dataclasses import dataclass
from .augmented import AugmentedState, State
from .base import IteratedFunctionBase, Parameters, Trajectory

__all__ = ['IteratedFunction']


Comparand = TypeVar('Comparand', bound=PyTree)
TheAugmentedState = TypeVar('TheAugmentedState', bound=AugmentedState[Any])


@dataclass
class IteratedFunction(IteratedFunctionBase[Parameters, State, Trajectory, TheAugmentedState],
                       Generic[Parameters, State, Comparand, Trajectory, TheAugmentedState]):
    """An IteratedFunction object models an iterated function.

    It is a generic class in terms of five generic types, all of which are pytrees:
        * Parameters, which models the iteration parameters,
        * State, which is the state at each iteration,
        * Comparand, which is calculated from state and compared between iterations to check
          convergence,
        * Trajectory, which is the type produced by sample_trajectory, and
        * TheAugmentedState, which is typically defined by one of the tjax subclasses.

    The two main methods of IteratedFunction are
    * find_fixed_point, which iterates until the fixed point is reached or other stopping conditions
      are met, and
    * sample_trajectory, which iterates for a fixed number of iterations and returns a trajectory.

    Args:
        minimum_iterations: The minimum number of iterations for this fixed point solver.  This must
            be positive.
        maximum_iterations: The maximum number of iterations for this fixed point solver.  This must
            be positive.
        rtol: The relative tolerance for the comparison stopping condition.
        atol: The absolute tolerance for the comparison stopping condition.
    """
    minimum_iterations: IntegralNumeric
    maximum_iterations: IntegralNumeric
    rtol: RealNumeric
    atol: RealNumeric

    # New methods ----------------------------------------------------------------------------------
    def find_fixed_point(self,
                         theta: Parameters,
                         initial_state: State) -> TheAugmentedState:
        """The fixed point finder.

        Args:
            theta: The parameters for which gradients can be calculated.
            initial_state: An initial guess of the final state.

        Returns:
            state: The augmented state at the fixed point.
        """
        def f(augmented: TheAugmentedState) -> TheAugmentedState:
            new_state = self.sampled_state(theta, augmented.current_state)
            return self.iterate_augmented(new_state, augmented)

        return while_loop(partial(self.state_needs_iteration, theta),
                          f,
                          self.initial_augmented(initial_state))

    def debug_fixed_point(self, theta: Parameters, initial_state: State) -> TheAugmentedState:
        """This method is identical to find_fixed_point, but avoids using while_loop."""
        augmented = self.initial_augmented(initial_state)
        while self.state_needs_iteration(theta, augmented):
            new_state = self.sampled_state(theta, augmented.current_state)
            augmented = self.iterate_augmented(new_state, augmented)
        return augmented

    def state_needs_iteration(self, theta: Parameters, augmented: TheAugmentedState) -> Array:
        """Whether the state needs to be iterated.

        Args:
            theta: The parameters.
            augmented: The state.
        Returns: True while iteration needs to continue.
        """
        enough_iterations = augmented.iterations >= self.minimum_iterations
        converged = self.converged(augmented)
        not_too_many_iterations = augmented.iterations < self.maximum_iterations
        not_converged_or_done = jnp.logical_not(
            jnp.logical_and(enough_iterations, converged))
        return jnp.logical_and(not_too_many_iterations,
                               not_converged_or_done)

    # Abstract methods -----------------------------------------------------------------------------
    def expected_state(self, theta: Parameters, state: State) -> State:
        """The expected value of the next state given the old one.

        This is used by the combinator.
        """
        raise NotImplementedError

    def converged(self, augmented: TheAugmentedState) -> BooleanNumeric:
        """A Boolean Array of shape () indicating whether the pytrees are close."""
        raise NotImplementedError

    def extract_comparand(self, state: State) -> Comparand:
        """Extracts the "comparand" from the state.

        This is a subset of the values in the state that are compared by subclasses to detect
        convergence:
        * In ZeroIteratedFunction, the comparand is compared with zero.
        * In ComparingIteratedFunction, the comparand is compared in successive states.
        * In StochasticIteratedFunction, the mean and variance of the comparand are compared.
        """
        raise NotImplementedError

    def minimum_tolerances(self, augmented: TheAugmentedState) -> tuple[RealNumeric, RealNumeric]:
        """The minimum tolerances that would lead to convergence now.

        Returns:
            The minimum value of atol that would lead to convergence now.
            The minimum value of rtol that would lead to convergence now.
        """
        raise NotImplementedError
