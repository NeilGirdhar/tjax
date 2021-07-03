from __future__ import annotations

from typing import Callable, Generic, Optional, Sequence, Tuple, TypeVar

import numpy as np
from jax.tree_util import tree_map
from matplotlib.axes import Axes

from .annotations import PyTree, RealArray, SliceLike
from .dataclasses import dataclass
from .leaky_integral import leaky_integrate_time_series

__all__ = ['PlottableTrajectory']


Trajectory = TypeVar('Trajectory', bound=PyTree)


@dataclass
class PlottableTrajectory(Generic[Trajectory]):
    trajectory: Trajectory
    "The trajectory is a PyTree containing the plotted data in its dynamic attributes."
    times: RealArray
    "The times corresponding to the data points in each of the plotted attributes."

    def plot(self,
             data: RealArray,
             axis: Axes,
             *,
             title: Optional[str] = None,
             legend: int = 0,
             labels: Optional[Sequence[str]] = None,
             simple_label: Optional[str] = None,
             label_function: Optional[Callable[[Tuple[int, ...]], str]] = None,
             decay: Optional[float] = None,
             clip_slice: SliceLike = slice(None),
             clip_boolean: Optional[RealArray] = None) -> None:
        """
        Plot the PlottableTrajectory into a matplotlib axis.
        Args:
            data: The data to be plotted.  Typically, this is a sub-element of trajectory.
            axis: A matplotlib axis to plot into.
            title: The title of the subplot.
            legend: The maximum number of entries for which to generate a legend.
            labels: The labels corresponding to each plot.
            simple_label: A single label with which to construct the labels.
            label_function: A function with which to construct the labels.
            decay: If not none, applies a leaky-integral with the given decay to the plotted points.
            clip_slice: Restrict the plot to a slice.
            clip_boolean: Restrict the plot to elements where this array is true.
        """
        if title is not None:
            axis.set_title(title)

        unclipped_times = self.times[clip_slice]
        all_ys = np.asarray(data)
        if not np.all(np.isfinite(all_ys)):
            print(f"Graph {'' if title is None else title + ' '}contains infinite numbers.")
            all_ys = np.where(np.isfinite(all_ys), all_ys, 0.0)

        plot_indices = list(np.ndindex(*all_ys.shape[1:]))
        if labels is not None:
            if len(plot_indices) != len(labels):
                raise ValueError
        elif simple_label is not None:
            if not plot_indices:
                labels = []
            elif len(plot_indices) == 1:
                labels = [simple_label]
            elif all_ys.ndim == 2:
                labels = [f"{simple_label} {x[0]}" for x in plot_indices]
            else:
                labels = [f"{simple_label} {x}" for x in plot_indices]
        elif label_function is not None:
            labels = [label_function(x) for x in plot_indices]
        else:
            labels = [str(x) for x in plot_indices]

        for plot_index, label in zip(plot_indices, labels):
            ys = all_ys[(clip_slice, *plot_index)]
            if decay is not None:
                ys = leaky_integrate_time_series(ys, decay)
            if clip_boolean is not None:
                ys = np.array(ys)
                ys[~clip_boolean] = np.nan
                times = np.array(unclipped_times)
                times[~clip_boolean] = np.nan
            else:
                times = unclipped_times
            axis.plot(times, ys, label=label)
        number_of_graph_lines = np.prod(data.shape[1:])
        if legend > 0 and 0 < number_of_graph_lines <= legend:
            axis.legend()

    def slice_into(self, s: SliceLike) -> Trajectory:
        "Return a new trajectory whose dynamic attributed are sliced."
        return tree_map(lambda x: x[s], self.trajectory)
