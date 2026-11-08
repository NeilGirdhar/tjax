"""Gradient transformations that retain the identity of each loss until the update."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, Self, override

import jax.numpy as jnp
from jax import tree

from tjax._src.annotations import PyTree, RealNumeric
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

    One instance handles a fixed group of losses on the same parameters.  Its state can hold
    statistics for individual losses and for the group.

    Args:
        loss_ids: The distinct IDs of the losses, in the order that the transformation uses them.
    """

    loss_ids: tuple[str, ...] = field(static=True, kw_only=True)

    @abstractmethod
    def init(self, parameters: Parameters) -> State:
        """Initialize state for the parameter pytree.

        Implementations validate ``loss_ids`` with ``check_loss_ids`` here rather than at
        construction, since JAX reconstructs the transformation each time it unflattens it.
        """
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        gradients_by_loss: Mapping[str, Parameters],
        state: State,
        parameters: Parameters,
    ) -> tuple[Parameters, State]:
        """Return the additive parameter update and the next transform state.

        The mapping has one gradient for each of ``loss_ids``, with the same pytree structure as
        the parameters.
        """
        raise NotImplementedError


@dataclass
class CombinedGradientState[Statistics: PyTree](GradientState):
    """Gradient state for :class:`GradientCombiner`.

    Args:
        inner_state: The state of the inner transformation.
        statistics: The statistics that the combiner keeps between steps.
    """

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
        loss_ids: The distinct IDs of the losses, in the order that :meth:`combine` receives their
            gradients.
    """

    inner: GradientTransformation[Any, Parameters]

    def init_statistics(self, parameters: Parameters) -> Statistics:
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
    def init(self, parameters: Parameters) -> CombinedGradientState[Statistics]:
        check_loss_ids(self.loss_ids)
        return CombinedGradientState[Statistics](
            self.inner.init(parameters), self.init_statistics(parameters)
        )

    @override
    def update(
        self,
        gradients_by_loss: Mapping[str, Parameters],
        state: CombinedGradientState[Statistics],
        parameters: Parameters,
    ) -> tuple[Parameters, CombinedGradientState[Statistics]]:
        check_gradients(gradients_by_loss, self.loss_ids)
        gradients = [gradients_by_loss[loss_id] for loss_id in self.loss_ids]
        gradient, statistics = self.combine(gradients, state.statistics)
        updates, inner_state = self.inner.update(gradient, state.inner_state, parameters)
        return updates, CombinedGradientState[Statistics](inner_state, statistics)


@dataclass
class AveragedGradientsState[Parameters: PyTree](GradientState):
    """Gradient state for :class:`AveragedGradients`.

    Args:
        averages: The decaying average of each loss's gradient.
        inner_state: The state of the inner transformation.
    """

    averages: Mapping[str, Parameters]
    inner_state: PyTree


@dataclass
class AveragedGradients[Parameters: PyTree](
    MultiLossGradientTransformation[AveragedGradientsState[Parameters], Parameters]
):
    """Pass a decaying average of each loss's gradient to a multi-loss transformation.

    Losses can arrive at different times, and a loss that hasn't arrived has a zero gradient, so a
    multi-loss transformation would otherwise never see the losses together.  Averaging keeps each
    arrival present for later ones to meet.  It also reduces noise, without which two gradient
    estimates in many dimensions have a cosine near zero whether or not their losses conflict.

    An average is zero until its loss first arrives.  The decay should keep an arrival present for
    longer than the gap between arrivals of the slowest loss.  ``MTAdam`` doesn't need this, since
    its first moments are already such averages.

    Build it with :meth:`create_from`, which takes the loss IDs from the inner transformation.

    Args:
        inner: The transformation applied to the averages.
        decay: The fraction of each average that carries over to the next step.
        loss_ids: The inner transformation's loss IDs.
    """

    inner: MultiLossGradientTransformation[Any, Parameters]
    decay: RealNumeric = 0.9

    @classmethod
    def create_from(
        cls, inner: MultiLossGradientTransformation[Any, Parameters], *, decay: RealNumeric = 0.9
    ) -> Self:
        """Average the gradients passed to ``inner``, using its loss IDs."""
        return cls(inner, decay, loss_ids=inner.loss_ids)

    @override
    def init(self, parameters: Parameters) -> AveragedGradientsState[Parameters]:
        check_loss_ids(self.loss_ids)
        if self.loss_ids != self.inner.loss_ids:
            msg = f"loss_ids {self.loss_ids} differ from the inner {self.inner.loss_ids}"
            raise ValueError(msg)
        averages = {loss_id: tree.map(jnp.zeros_like, parameters) for loss_id in self.loss_ids}
        return AveragedGradientsState[Parameters](averages, self.inner.init(parameters))

    @override
    def update(
        self,
        gradients_by_loss: Mapping[str, Parameters],
        state: AveragedGradientsState[Parameters],
        parameters: Parameters,
    ) -> tuple[Parameters, AveragedGradientsState[Parameters]]:
        check_gradients(gradients_by_loss, self.loss_ids)
        averages = {
            loss_id: tree.map(
                lambda average, g: self.decay * average + (1.0 - self.decay) * g,
                state.averages[loss_id],
                gradients_by_loss[loss_id],
            )
            for loss_id in self.loss_ids
        }
        updates, inner_state = self.inner.update(averages, state.inner_state, parameters)
        return updates, AveragedGradientsState[Parameters](averages, inner_state)
