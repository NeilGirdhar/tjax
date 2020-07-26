from __future__ import annotations

from functools import partial
from typing import Any, Callable, Generic, Optional, Tuple, TypeVar

from jax import jit
from jax import numpy as jnp
from jax.experimental.host_callback import id_tap
from jax.lax import scan, while_loop
from jax.tree_util import tree_map, tree_multimap

from tjax import PyTree, Tensor, dataclass, default_atol, default_rtol

from .augmented import AugmentedState, State

__all__ = ['IteratedFunction']


Parameters = TypeVar('Parameters', bound=PyTree)
Trajectory = TypeVar('Trajectory', bound=PyTree)
TheAugmentedState = TypeVar('TheAugmentedState', bound=AugmentedState[Any])


# https://github.com/python/mypy/issues/8539
@dataclass  # type: ignore
class IteratedFunction(Generic[Parameters, State, TheAugmentedState]):
    """
    An IteratedFunction object models an iterated function.

    It is a generic class in terms of two generic types:
        * Parameters, which models a PyTree subtype for parameters, and
        * State, which models a PyTree subtype for the state.

    The minimum requirement is to implement iterate_state, which iterates the state given
    parameters.

    In the case of a stochastic iterated function, iterate_augmented should be augmented, and should
    return the next augmented state.  iterate_state should return the expected value of the next
    stochastic state.

    It is possible to modify the stopping condition by overriding state_needs_iteration.

    The two main methods of IteratedFunction are
    * find_fixed_point, which iterates until the fixed point is reached or other stopping conditions
      are met, and
    * sample_trajectory, which iterates for a fixed number of iterations and returns a trajectory.

    Args:
        iteration_limit: The maximum number of iterations for this fixed point solver.  This must be
            positive.
        rtol: The relative tolerance for the comparison stopping condition.
        atol: The absolute tolerance for the comparison stopping condition.
        initial_rng: The inital random number generator.
    """

    minimum_iterations: int = 10
    iteration_limit: int
    rtol: float = default_rtol
    atol: float = default_atol

    # New methods ----------------------------------------------------------------------------------
    @jit
    def find_fixed_point(self, theta: Parameters, initial_state: State) -> TheAugmentedState:
        """
        Args:
            theta: The parameters for which gradients can be calculated.
            initial_state: An initial guess of the final state.
        Returns:
            state: The augmented state at the fixed point.
        """
        return while_loop(partial(self.state_needs_iteration, theta),
                          partial(self.iterate_augmented, theta),
                          self.initial_state(initial_state))

    @partial(jit, static_argnums=(3, 4, 5))
    def sample_trajectory(self,
                          theta: Parameters,
                          initial_state: State,
                          iteration_limit: int,
                          tap_function: Optional[Callable[[None], None]],
                          extract: Callable[[TheAugmentedState], Trajectory]) -> (
                              Tuple[TheAugmentedState, Trajectory]):
        """
        Args:
            theta: The parameters for which gradients can be calculated.
            initial_state: An initial guess of the final state.
            iteration_limit: The number of steps in the trajectory.  Unlike the eponymous member
                variable, this must be static.
        Returns:
            x_star: The augmented state at the fixed point.
            trajectory: A PyTree representing the trajectory of states.
        """
        def f(augmented: TheAugmentedState, x: None) -> Tuple[
                TheAugmentedState, Trajectory]:
            new_augmented = self.iterate_augmented(theta, augmented)
            extracted = extract(new_augmented)
            if tap_function is not None:
                extracted = id_tap(tap_function, None, result=extracted)
            return new_augmented, extracted
        return scan(f, self.initial_state(initial_state), None, iteration_limit)

    def debug_fixed_point(self, theta: Parameters, initial_state: State) -> TheAugmentedState:
        """
        This method is identical to find_fixed_point, but avoids using while_loop and its
        concomitant jit.
        """
        augmented = self.initial_state(initial_state)
        while self.state_needs_iteration(theta, augmented):
            augmented = self.iterate_augmented(theta, augmented)
        return augmented

    def debug_trajectory(self,
                         theta: Parameters,
                         initial_state: State,
                         iteration_limit: int,
                         tap_function: Optional[Callable[[None], None]],
                         extract: Callable[[TheAugmentedState], Trajectory]) -> (
                             Tuple[TheAugmentedState, Trajectory]):
        """
        This method is identical to sample_trajectory, but avoids using scan and its concomitant
        jit.
        """
        augmented = self.initial_state(initial_state)
        trajectory = tree_map(lambda x: jnp.zeros((0,) + x.shape), extract(augmented))
        for _ in range(iteration_limit):
            augmented = self.iterate_augmented(theta, augmented)
            extracted = extract(augmented)
            trajectory = tree_multimap(jnp.append, trajectory, extracted)
            if tap_function is not None:
                tap_function(None)
        return augmented, trajectory

    def state_needs_iteration(self, theta: Parameters, augmented: TheAugmentedState) -> bool:
        """
        Args:
            theta: The parameters.
            augmented: The state.
        Returns: True while iteration needs to continue.
        """
        converged = jnp.logical_not(jnp.logical_and(augmented.iterations > self.minimum_iterations,
                                                    self.converged(augmented)))
        too_many_iterations = augmented.iterations < self.iteration_limit
        return jnp.logical_and(too_many_iterations, converged)

    # Abstract methods -----------------------------------------------------------------------------
    def initial_state(self, initial_state: State) -> TheAugmentedState:
        raise NotImplementedError

    def iterate_state(self, theta: Parameters, x: State) -> State:
        """
        Returns: The expected value of the next state given the old one.  A state must be
            differentiable.
        """
        raise NotImplementedError

    def iterate_augmented(self,
                          theta: Parameters,
                          augmented: TheAugmentedState) -> TheAugmentedState:
        """
        Returns: The value of the next augmented state given the old one.
        """
        raise NotImplementedError

    def converged(self, augmented: TheAugmentedState) -> Tensor:
        """
        Returns: A Boolean Tensor of shape () indicating whether the pytrees are close.
        """
        raise NotImplementedError
