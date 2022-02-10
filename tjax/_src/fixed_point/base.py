from __future__ import annotations

from typing import Callable, Generic, Optional, Tuple, TypeVar

from jax.experimental.host_callback import id_tap
from jax.lax import scan

from ..annotations import PyTree, TapFunctionTransforms
from ..dataclasses import dataclass
from .augmented import State

__all__ = ['IteratedFunctionBase']


Parameters = TypeVar('Parameters', bound=PyTree)
Trajectory = TypeVar('Trajectory', bound=PyTree)
TheAugmentedState = TypeVar('TheAugmentedState')
TapFunction = Callable[[None, TapFunctionTransforms], None]


@dataclass
class IteratedFunctionBase(Generic[Parameters, State, Trajectory, TheAugmentedState]):
    """
    An IteratedFunctionBase object models any iterated function.

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
                trajectory = id_tap(tap_function,  # type: ignore[no-untyped-call]
                                    None, result=trajectory)
            return new_augmented, trajectory
        return scan(f, self.initial_augmented(initial_state), None, maximum_iterations)

    # Abstract methods -----------------------------------------------------------------------------
    def initial_augmented(self, initial_state: State) -> TheAugmentedState:
        raise NotImplementedError

    def sampled_state(self, theta: Parameters, state: State) -> State:
        """
        Returns: A sampled value of the next state in a trajectory.  This is used when finding the
            fixed point.  It is included in this base class because it is nearly always called by
            sampled_state_trajectory.
        """
        raise NotImplementedError

    def sampled_state_trajectory(self,
                                 theta: Parameters,
                                 augmented: TheAugmentedState) -> Tuple[State, Trajectory]:
        """
        Returns:
            sampled_state: A sampled value of the next state in a trajectory.
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
