from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

from jax.experimental import io_callback
from jax.lax import scan

from tjax.dataclasses import dataclass

from ..annotations import PyTree
from .augmented import State

Parameters = TypeVar('Parameters', bound=PyTree)
Trajectory = TypeVar('Trajectory', bound=PyTree)
TheAugmentedState = TypeVar('TheAugmentedState')


@dataclass
class IteratedFunctionBase(Generic[Parameters, State, Trajectory, TheAugmentedState]):
    """An IteratedFunctionBase object models any iterated function.

    It is a generic class in terms of four generic types, all of which are pytrees:
        * Parameters, which models the iteration parameters,
        * State, which is the state at each iteration,
        * Trajectory, which is the type produced by sample_trajectory, and
        * TheAugmentedState, which is typically defined by one of the tjax subclasses.
    """
    # New methods ----------------------------------------------------------------------------------
    def sample_trajectory(self,
                          theta: Parameters,
                          initial_state: State,
                          maximum_iterations: int,
                          callback: Callable[..., None] | None
                          ) -> tuple[TheAugmentedState, Trajectory]:
        """Sample the next augmented state in a trajectory and information about the trajectory.

        Args:
            theta: The parameters for which gradients can be calculated.
            initial_state: An initial guess of the final state.
            maximum_iterations: The number of steps in the trajectory.  Unlike the eponymous member
                variable, this must be static.
            callback: A function that will be called every iteration.

        Returns:
            x_star: The augmented state at the fixed point.
            trajectory: A PyTree representing the trajectory of states.
        """
        def f(augmented: TheAugmentedState, x: None) -> tuple[TheAugmentedState, Trajectory]:
            trajectory: Trajectory
            new_state, trajectory = self.sampled_state_trajectory(theta, augmented)
            new_augmented = self.iterate_augmented(new_state, augmented)
            if callback is not None:
                io_callback(callback, None)
            return new_augmented, trajectory
        return scan(f, self.initial_augmented(initial_state), None, maximum_iterations)

    # Abstract methods -----------------------------------------------------------------------------
    def initial_augmented(self, initial_state: State) -> TheAugmentedState:
        raise NotImplementedError

    def sampled_state(self, theta: Parameters, state: State) -> State:
        """Sample the next state in a trajectory.

        Returns: A sampled value of the next state in a trajectory.  This is used when finding the
            fixed point.  It is included in this base class because it is nearly always called by
            sampled_state_trajectory.
        """
        raise NotImplementedError

    def sampled_state_trajectory(self,
                                 theta: Parameters,
                                 augmented: TheAugmentedState
                                 ) -> tuple[State, Trajectory]:
        """Sample the next state in a trajectory and information about the trajectory.

        Returns:
            sampled_state: A sampled value of the next state in a trajectory.
            trajectory: A value to be concatenated into a trajectory.
        """
        raise NotImplementedError

    def iterate_augmented(self,
                          new_state: State,
                          augmented: TheAugmentedState
                          ) -> TheAugmentedState:
        """Fold the state into the augmented state.

        Args:
            new_state: The new state to fold into the augmented state.
            augmented: The last augmented state.
        Returns: The next augmented state.
        """
        raise NotImplementedError
