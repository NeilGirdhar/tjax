from __future__ import annotations

from typing import Generic, Optional, TypeVar, Union

import numpy as np
from chex import Array
from jax.tree_util import tree_map
from matplotlib.axes import Axes

from .annotations import PyTree
from .dataclass import dataclass

__all__ = ['PlottableTrajectory']


Trajectory = TypeVar('Trajectory', bound=PyTree)


@dataclass
class PlottableTrajectory(Generic[Trajectory]):
    trajectory: Trajectory
    "The trajectory is PyTree containing the plotted data in its nonstatic attributes."
    iterations: int
    "The number of data points in each of the plotted attributes."
    time_step: Array
    "The period between adjacent data points."

    def plot(self,
             attribute_name: str,
             axis: Axes,
             subplot_title: str,
             *,
             legend: int = 0,
             decay: Optional[float] = None,
             clip_slice: slice = slice(None)) -> None:
        """
        Plot the PlottableTrajectory into a matplotlib axis.
        Args:
            attribute_name: The name of the attribute in the trajectory to plot.
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
        all_ys = np.asarray(getattr(self.trajectory, attribute_name))
        if not np.all(np.isfinite(all_ys)):
            print(f"Graph {attribute_name} contains infinite numbers.")

        for plot_index in np.ndindex(all_ys.shape[1:]):
            ys = all_ys[(clip_slice, *plot_index)]
            if decay is not None:
                new_ys = np.zeros_like(ys)
                acc = np.zeros_like(ys[0])
                denominator = 0.0
                scale = np.exp(-decay * self.time_step)
                for j, y in enumerate(ys):
                    acc = acc * scale + y * self.time_step
                    denominator = denominator * scale + self.time_step
                    new_ys[j] = acc / denominator
                ys = new_ys
            axis.plot(xs, ys, label=str(plot_index))
        if legend > 0 and self.number_of_graph_lines(attribute_name) <= legend:
            axis.legend()

    def number_of_graph_lines(self, attribute_name: str) -> int:
        """
        Args:
            attribute_name: The name of the attribute in the trajectory.
        Returns: The number of graphed lines for the attribute.
        """
        return np.prod(getattr(self.trajectory, attribute_name).shape[1:])

    def slice_into(self, s: Union[int, slice]) -> Trajectory:
        "Return a new trajectory whose nonstatic attributed are sliced."
        return tree_map(lambda x: x[s], self.trajectory)
