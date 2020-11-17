from __future__ import annotations

from typing import Generic, Optional, TypeVar, Union

import numpy as np
from chex import Array
from jax.tree_util import tree_map
from matplotlib.axes import Axes

from .annotations import PyTree
from .dataclass import dataclass
from .leaky_integral import leaky_integrate_time_series

__all__ = ['PlottableTrajectory']


Trajectory = TypeVar('Trajectory', bound=PyTree)


@dataclass
class PlottableTrajectory(Generic[Trajectory]):
    trajectory: Trajectory
    "The trajectory is PyTree containing the plotted data in its nonstatic attributes."
    iterations: int
    "The number of data points in each of the plotted attributes."
    time_step: float
    "The period between adjacent data points."

    def plot(self,
             data: Array,
             axis: Axes,
             subplot_title: str,
             *,
             legend: int = 0,
             decay: Optional[float] = None,
             clip_slice: slice = slice(None)) -> None:
        """
        Plot the PlottableTrajectory into a matplotlib axis.
        Args:
            data: The data to be plotted.  Typically, this is a sub-element of trajectory.
            axis: A matplotlib axis to plot into.
            subplot_title: The title of the subplot.
            legend: The maximum number of entries for which to generate a legend.
            decay: If not none, applies a leaky-integral with the given decay to the plotted points.
            clip_slice: Restrict the plot to a slice.
        """
        axis.set_title(subplot_title)
        axis.ticklabel_format(useOffset=False)

        xs = np.linspace(0.0, self.iterations * self.time_step, self.iterations, endpoint=False)[
            clip_slice]
        all_ys = np.asarray(data)
        if not np.all(np.isfinite(all_ys)):
            print(f"Graph {subplot_title} contains infinite numbers.")

        for plot_index in np.ndindex(all_ys.shape[1:]):
            ys = all_ys[(clip_slice, *plot_index)]
            if decay is not None:
                ys = leaky_integrate_time_series(ys, decay)
            axis.plot(xs, ys, label=str(plot_index))
        number_of_graph_lines = np.prod(data.shape[1:])
        if legend > 0 and 0 < number_of_graph_lines <= legend:
            axis.legend()

    def slice_into(self, s: Union[int, slice]) -> Trajectory:
        "Return a new trajectory whose nonstatic attributed are sliced."
        return tree_map(lambda x: x[s], self.trajectory)
