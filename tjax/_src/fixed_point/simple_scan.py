from __future__ import annotations

from typing import Generic

from typing_extensions import override

from ..dataclasses import dataclass
from .augmented import State
from .base import IteratedFunctionBase, Parameters, Trajectory

__all__ = ['SimpleScan']


@dataclass
class SimpleScan(IteratedFunctionBase[Parameters, State, Trajectory, State],
                 Generic[Parameters, State, Trajectory]):
    """A SimpleScan object models an iterated function that runs for a fixed number of steps.

    It is a generic class in terms of three generic types, all of which are pytrees:
        * Parameters, which models the iteration parameters,
        * State, which is the state at each iteration,
        * Trajectory, which is the type produced by sample_trajectory, and

    The main method of SimpleScan is sample_trajectory, which iterates for a fixed number of
    iterations and returns a trajectory.
    """
    # Implemented methods --------------------------------------------------------------------------
    @override
    def initial_augmented(self, initial_state: State) -> State:
        return initial_state

    @override
    def iterate_augmented(self,
                          new_state: State,
                          augmented: State) -> State:
        return new_state
