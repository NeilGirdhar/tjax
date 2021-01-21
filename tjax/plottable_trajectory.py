from __future__ import annotations

from typing import Generic, Optional, Sequence, TypeVar, Union

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
    "The trajectory is a PyTree containing the plotted data in its nonstatic attributes."
    times: np.ndarray
    "The times corresponding to the data points in each of the plotted attributes."

    def plot(self,
             data: Array,
             axis: Axes,
             *,
             title: Optional[str] = None,
             legend: int = 0,
             labels: Optional[Sequence[str]] = None,
             decay: Optional[float] = None,
             clip_slice: slice = slice(None)) -> None:
        """
        Plot the PlottableTrajectory into a matplotlib axis.
        Args:
            data: The data to be plotted.  Typically, this is a sub-element of trajectory.
            axis: A matplotlib axis to plot into.
            title: The title of the subplot.
            legend: The maximum number of entries for which to generate a legend.
            decay: If not none, applies a leaky-integral with the given decay to the plotted points.
            clip_slice: Restrict the plot to a slice.
        """
        if title is not None:
            axis.set_title(title)
        axis.ticklabel_format(useOffset=False)

        times = self.times[clip_slice]
        all_ys = np.asarray(data)
        if not np.all(np.isfinite(all_ys)):
            print(f"Graph {'' if title is None else title + ' '}contains infinite numbers.")
            all_ys = np.where(np.isfinite(all_ys), all_ys, 0.0)

        plot_indices = list(np.ndindex(all_ys.shape[1:]))
        if labels is None:
            labels = [str(x) for x in plot_indices]
        for plot_index, label in zip(plot_indices, labels):
            ys = all_ys[(clip_slice, *plot_index)]
            if decay is not None:
                ys = leaky_integrate_time_series(ys, decay)
            axis.plot(times, ys, label=label)
        number_of_graph_lines = np.prod(data.shape[1:])
        if labels is not None or legend > 0 and 0 < number_of_graph_lines <= legend:
            axis.legend()

    def slice_into(self, s: Union[int, slice]) -> Trajectory:
        "Return a new trajectory whose nonstatic attributed are sliced."
        return tree_map(lambda x: x[s], self.trajectory)
