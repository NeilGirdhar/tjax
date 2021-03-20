from __future__ import annotations

from functools import partial
from typing import Any, Callable, Generic, Optional, Tuple, TypeVar

import jax.numpy as jnp
from chex import Array
from jax import jit
from jax.experimental.host_callback import id_tap
from jax.lax import scan, while_loop
from jax.tree_util import tree_multimap

from ..annotations import PyTree, TapFunctionTransforms
from ..dataclass import dataclass, field
from ..dtypes import default_atol, default_rtol
from .augmented import AugmentedState, State

__all__ = ['IteratedFunction']


Parameters = TypeVar('Parameters', bound=PyTree)
Comparand = TypeVar('Comparand', bound=PyTree)
Trajectory = TypeVar('Trajectory', bound=PyTree)
TheAugmentedState = TypeVar('TheAugmentedState', bound=AugmentedState[Any])
TapFunction = Callable[[None, TapFunctionTransforms], None]


# https://github.com/python/mypy/issues/8539
@dataclass  # type: ignore
class IteratedFunction(Generic[Parameters, State, Comparand, Trajectory, TheAugmentedState]):
    """
    An IteratedFunction object models an iterated function.

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

    minimum_iterations: Array
    maximum_iterations: Array
    rtol: float = field(default_factory=default_rtol)
    atol: float = field(default_factory=default_atol)

    # New methods ----------------------------------------------------------------------------------
    def find_fixed_point(self,  # pylint: disable=function-redefined, method-hidden
                         theta: Parameters,
                         initial_state: State) -> TheAugmentedState:
        """
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

    find_fixed_point = jit(find_fixed_point)  # pylint: disable=used-before-assignment

    def sample_trajectory(self,  # pylint: disable=function-redefined, method-hidden
                          theta: Parameters,
                          initial_state: State,
                          maximum_iterations: int,
                          tap_function: Optional[TapFunction]) -> (
                              Tuple[TheAugmentedState, Trajectory]):
        """
        Args:
            theta: The parameters for which gradients can be calculated.
            initial_state: An initial guess of the final state.
            maximum_iterations: The number of steps in the trajectory.  Unlike the eponymous member
                variable, this must be static.
        Returns:
            x_star: The augmented state at the fixed point.
            trajectory: A PyTree representing the trajectory of states.
        """
        def f(augmented: TheAugmentedState, x: None) -> Tuple[TheAugmentedState, Trajectory]:
            trajectory: Trajectory
            new_state, trajectory = self.sampled_state_trajectory(theta, augmented)
            new_augmented = self.iterate_augmented(new_state, augmented)
            if tap_function is not None:
                trajectory = id_tap(tap_function, None, result=trajectory)
            return new_augmented, trajectory
        return scan(f, self.initial_augmented(initial_state), None, maximum_iterations)

    # pylint: disable=used-before-assignment
    sample_trajectory = jit(sample_trajectory, static_argnums=(3, 4))

    def debug_fixed_point(self, theta: Parameters, initial_state: State) -> TheAugmentedState:
        """
        This method is identical to find_fixed_point, but avoids using while_loop and its
        concomitant jit.
        """
        augmented = self.initial_augmented(initial_state)
        while self.state_needs_iteration(theta, augmented):
            new_state = self.sampled_state(theta, augmented.current_state)
            augmented = self.iterate_augmented(new_state, augmented)
        return augmented

    def debug_trajectory(self,
                         theta: Parameters,
                         initial_state: State,
                         maximum_iterations: int,
                         tap_function: Optional[TapFunction]) -> (
                             Tuple[TheAugmentedState, Trajectory]):
        """
        This method is identical to sample_trajectory, but avoids using scan and its concomitant
        jit.
        """
        augmented = self.initial_augmented(initial_state)
        for i in range(maximum_iterations):
            trajectory: Trajectory
            concatenated_trajectory: Trajectory
            new_state, trajectory = self.sampled_state_trajectory(theta, augmented)
            augmented = self.iterate_augmented(new_state, augmented)
            concatenated_trajectory = (
                trajectory
                if i == 0
                else tree_multimap(jnp.append, concatenated_trajectory, trajectory))  # noqa: F821
            if tap_function is not None:
                tap_function(None, ())
        return augmented, concatenated_trajectory

    def state_needs_iteration(self, theta: Parameters, augmented: TheAugmentedState) -> bool:
        """
        Args:
            theta: The parameters.
            augmented: The state.
        Returns: True while iteration needs to continue.
        """
        enough_iterations = augmented.iterations >= self.minimum_iterations
        converged = self.converged(augmented)
        not_too_many_iterations = augmented.iterations < self.maximum_iterations
        return jnp.logical_and(not_too_many_iterations,
                               jnp.logical_not(jnp.logical_and(enough_iterations, converged)))

    # Abstract methods -----------------------------------------------------------------------------
    def initial_augmented(self, initial_state: State) -> TheAugmentedState:
        raise NotImplementedError

    def expected_state(self, theta: Parameters, state: State) -> State:
        """
        Returns: The expected value of the next state given the old one.  This is used by the
            combinator.
        """
        raise NotImplementedError

    def sampled_state(self, theta: Parameters, state: State) -> State:
        """
        Returns: A sampled value of the next state in a trajectory.  This is used when finding the
            fixed point.
        """
        raise NotImplementedError

    def sampled_state_trajectory(self,
                                 theta: Parameters,
                                 augmented: TheAugmentedState) -> Tuple[State, Trajectory]:
        """
        Returns:
            sampled_state: A sampled value of the next state in a trajectory.  This is used when
                finding the fixed point.
            trajectory: A value to be concatenated into a trajectory.
        """
        raise NotImplementedError

    def iterate_augmented(self,
                          new_state: State,
                          augmented: TheAugmentedState) -> TheAugmentedState:
        """
        Args:
            new_state: The new state to fold into the augmented state.
            augmented: The last augmented state.
        Returns: The next augmented state.
        """
        raise NotImplementedError

    def converged(self, augmented: TheAugmentedState) -> Array:
        """
        Returns: A Boolean Array of shape () indicating whether the pytrees are close.
        """
        raise NotImplementedError

    def extract_comparand(self, state: State) -> Comparand:
        """
        Returns: The "comparand", which is a subset of the values in the state that are compared by
            subclasses to detect convergence.  In ComparingIteratedFunction, the comparand will be
            compared in successive states to detect convergence.  In StochasticIteratedFunction, the
            mean and variance of the comparand are used to detect convergence.
        """
        raise NotImplementedError

    def minimum_tolerances(self, augmented: TheAugmentedState) -> Tuple[Array, Array]:
        """
        Returns:
            The minimum value of atol that would lead to convergence now.
            The minimum value of rtol that would lead to convergence now.
        """
        raise NotImplementedError
