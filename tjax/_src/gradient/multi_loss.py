"""Gradient transformations that retain the identity of each loss until the update."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, override

from tjax._src.annotations import PyTree
from tjax.dataclasses import dataclass, field

from .transform import GradientState, GradientTransformation


def check_loss_ids(loss_ids: tuple[str, ...]) -> None:
    """Raise an error unless the loss IDs are nonempty and distinct."""
    if not loss_ids or len(set(loss_ids)) != len(loss_ids):
        msg = f"loss_ids must be nonempty and distinct: {loss_ids}"
        raise ValueError(msg)


def check_gradients(gradients_by_loss: Mapping[str, Any], loss_ids: tuple[str, ...]) -> None:
    """Raise an error unless there is one gradient for each loss ID."""
    if set(gradients_by_loss) != set(loss_ids):
        msg = f"Expected gradients for {loss_ids}, got {tuple(gradients_by_loss)}"
        raise ValueError(msg)


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


@dataclass
class CombinedGradientState[Statistics: PyTree](GradientState):
    """Gradient state for :class:`GradientCombiner`.

    Args:
        loss_ids: The loss IDs, in the order that ``combine`` receives their gradients.
        inner_state: The state of the inner transformation.
        statistics: The statistics that the combiner keeps between steps.
    """

    loss_ids: tuple[str, ...] = field(static=True)
    inner_state: PyTree
    statistics: Statistics


@dataclass
class GradientCombiner[Statistics: PyTree, Parameters: PyTree](
    MultiLossGradientTransformation[CombinedGradientState[Statistics], Parameters]
):
    """Combine the gradients of several losses, and pass the result to a transformation.

    Many methods for balancing losses only decide how to combine gradients, and leave the optimizer
    to the user.  Subclasses implement that decision in :meth:`combine`, and the inner
    transformation turns the combined gradient into a parameter update.

    Args:
        inner: The transformation applied to the combined gradient, such as ``Adam(1e-3)``.
    """

    inner: GradientTransformation[Any, Parameters]

    def init_statistics(self, parameters: Parameters, loss_ids: tuple[str, ...]) -> Statistics:
        """Return the statistics that :meth:`combine` keeps between steps.

        Subclasses without statistics can keep this default, which returns ``None``.
        """
        return None  # ty: ignore[invalid-return-type]

    @abstractmethod
    def combine(
        self, gradients: Sequence[Parameters], statistics: Statistics
    ) -> tuple[Parameters, Statistics]:
        """Return the combined gradient and the next statistics.

        Args:
            gradients: The gradient of each loss, in the order of the loss IDs.
            statistics: The current statistics.
        """
        raise NotImplementedError

    @override
    def init(
        self, parameters: Parameters, loss_ids: tuple[str, ...]
    ) -> CombinedGradientState[Statistics]:
        check_loss_ids(loss_ids)
        return CombinedGradientState[Statistics](
            loss_ids, self.inner.init(parameters), self.init_statistics(parameters, loss_ids)
        )

    @override
    def update(
        self,
        gradients_by_loss: Mapping[str, Parameters],
        state: CombinedGradientState[Statistics],
        parameters: Parameters,
    ) -> tuple[Parameters, CombinedGradientState[Statistics]]:
        check_gradients(gradients_by_loss, state.loss_ids)
        gradients = [gradients_by_loss[loss_id] for loss_id in state.loss_ids]
        gradient, statistics = self.combine(gradients, state.statistics)
        updates, inner_state = self.inner.update(gradient, state.inner_state, parameters)
        return updates, CombinedGradientState[Statistics](state.loss_ids, inner_state, statistics)
