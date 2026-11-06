"""Gradient transformations that retain the identity of each loss until the update."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

from tjax._src.annotations import PyTree
from tjax.dataclasses import dataclass

from .transform import GradientState


@dataclass
class MultiLossGradientTransformation[State: GradientState, Parameters: PyTree](ABC):
    """Combine gradients from several losses into one additive parameter update.

    One instance handles a group of losses on the same parameters. Its state can hold
    statistics for individual losses and for the group.
    """

    @abstractmethod
    def init(self, parameters: Parameters, loss_ids: tuple[str, ...]) -> State:
        """Initialize state for the parameter pytree and its loss group."""
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        gradients_by_loss: Mapping[str, Parameters],
        state: State,
        parameters: Parameters,
    ) -> tuple[Parameters, State]:
        """Return the additive parameter update and the next transform state.

        Each gradient has the same pytree structure as the parameters. The mapping
        uses the loss IDs supplied at initialization.
        """
        raise NotImplementedError
