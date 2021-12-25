from __future__ import annotations

from typing import Generic, TypeVar

from .annotations import PyTree, RealArray
from .dataclasses import dataclass

__all__ = ['PlottableTrajectory']


Trajectory = TypeVar('Trajectory', bound=PyTree)


@dataclass
class PlottableTrajectory(Generic[Trajectory]):
    trajectory: Trajectory
    "The trajectory is a PyTree containing the plotted data in its dynamic attributes."
    times: RealArray
    "The times corresponding to the data points in each of the plotted attributes."
