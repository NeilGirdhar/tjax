"""The MTAdam optimizer, which balances several losses automatically."""

from __future__ import annotations

from collections.abc import Mapping
from functools import reduce
from typing import override

import jax.numpy as jnp
import optax
from jax import tree

from tjax._src.annotations import JaxArray, JaxIntegralArray, PyTree, RealNumeric
from tjax._src.math_tools import abs_square
from tjax.dataclasses import dataclass

from .multi_loss import MultiLossGradientTransformation, check_gradients, check_loss_ids
from .schedule import ScalarOrSchedule, Schedule
from .transform import GradientState


@dataclass
class MTAdamState[Parameters: PyTree](GradientState):
    """Gradient state for :class:`MTAdam`.

    Args:
        count: The number of updates so far.
        first_moments: The running mean of each loss's normalized gradient.
        second_moments: The running mean of each loss's squared normalized gradient.
        gradient_norms: The running norm of each loss's gradient, with one scalar for each leaf of
            the parameters.
    """

    count: JaxIntegralArray
    first_moments: Mapping[str, Parameters]
    second_moments: Mapping[str, Parameters]
    gradient_norms: Mapping[str, PyTree]


@dataclass
class MTAdam[Parameters: PyTree](
    MultiLossGradientTransformation[MTAdamState[Parameters], Parameters]
):
    """Adam for several losses, balanced automatically.

    Summing losses with hand-tuned weights requires a search over those weights, and a single
    weight can't suit every layer or stay right as training progresses.  MTAdam balances the losses
    itself, so that they contribute equally at each leaf of the parameters and at each step.

    The first loss in ``loss_ids`` is the anchor.  Every loss's gradient is rescaled to the anchor's
    running norm, so the balanced gradients follow the anchor's magnitude over time.  A constant
    factor on any loss has no effect, because the Adam step is invariant to a constant gradient
    scale.  The anchor matters through how its magnitude changes: as its gradient shrinks relative
    to its history, the steps shrink, as they would under Adam on the anchor loss alone.

    References:
        Malkiel and Wolf, 2020,
            [MTAdam: Automatic Balancing of Multiple Training Loss
            Terms](https://arxiv.org/abs/2006.14683)

    Args:
        learning_rate: Global step-size schedule or scalar.
        b1: Decay rate for the first moment of each loss's gradient.
        b2: Decay rate for the second moment of each loss's gradient.
        b3: Decay rate for the running norm of each loss's gradient.
        eps: Regularization term for numerical stability.
        min_norm: The running norm below which a loss's gradient is left unscaled, and can't be
            the anchor.
        loss_ids: The distinct IDs of the losses, with the anchor loss first.
    """

    learning_rate: ScalarOrSchedule
    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.999
    b3: RealNumeric = 0.9
    eps: RealNumeric = 1e-8
    min_norm: RealNumeric = 1e-10

    @override
    def init(self, parameters: Parameters) -> MTAdamState[Parameters]:
        """Initialize state for the parameter pytree.

        Args:
            parameters: The weights.
        """
        loss_ids = self.loss_ids
        check_loss_ids(loss_ids)
        return MTAdamState[Parameters](
            jnp.zeros((), jnp.int32),
            {loss_id: tree.map(jnp.zeros_like, parameters) for loss_id in loss_ids},
            {loss_id: tree.map(jnp.zeros_like, parameters) for loss_id in loss_ids},
            {
                loss_id: tree.map(lambda x: jnp.zeros((), jnp.real(x).dtype), parameters)
                for loss_id in loss_ids
            },
        )

    @override
    def update(
        self,
        gradients_by_loss: Mapping[str, Parameters],
        state: MTAdamState[Parameters],
        parameters: Parameters,
    ) -> tuple[Parameters, MTAdamState[Parameters]]:
        """Return the additive parameter update and the next transform state.

        At each leaf, each loss's gradient is rescaled so that its running norm equals the anchor's.
        The anchor is the first loss whose running norm exceeds ``min_norm``, so a loss with a zero
        gradient can't zero out the others.  The running norm starts at the first gradient's norm,
        as in the authors' code, since starting it at one, as in the paper, would leave the first
        steps almost unbalanced.

        The update sums the bias-corrected first moments of the losses.  They share a denominator
        based on the largest second moment, so that the step is small when any loss is uncertain.

        Args:
            gradients_by_loss: The gradient of each loss, keyed by ``loss_ids``.
            state: The gradient state.
            parameters: The weights.
        """
        loss_ids = self.loss_ids
        check_gradients(gradients_by_loss, loss_ids)

        def running_norm(norm: JaxArray, gradient: JaxArray) -> JaxArray:
            current = jnp.linalg.vector_norm(gradient)
            return jnp.where(state.count == 0, current, self.b3 * norm + (1.0 - self.b3) * current)

        def anchor_norm(*norms: JaxArray) -> JaxArray:
            stacked = jnp.stack(norms)
            return stacked[jnp.argmax(stacked > self.min_norm)]

        def normalize(gradient: JaxArray, norm: JaxArray, anchor: JaxArray) -> JaxArray:
            usable = norm > self.min_norm
            return jnp.where(usable, anchor / jnp.where(usable, norm, 1.0), 1.0) * gradient

        norms = {
            loss_id: tree.map(
                running_norm, state.gradient_norms[loss_id], gradients_by_loss[loss_id]
            )
            for loss_id in loss_ids
        }
        anchors = tree.map(anchor_norm, *norms.values())
        normalized = {
            loss_id: tree.map(normalize, gradients_by_loss[loss_id], norms[loss_id], anchors)
            for loss_id in loss_ids
        }
        first_moments = {
            loss_id: tree.map(
                lambda m, g: self.b1 * m + (1.0 - self.b1) * g,
                state.first_moments[loss_id],
                normalized[loss_id],
            )
            for loss_id in loss_ids
        }
        second_moments = {
            loss_id: tree.map(
                lambda v, g: self.b2 * v + (1.0 - self.b2) * abs_square(g),
                state.second_moments[loss_id],
                normalized[loss_id],
            )
            for loss_id in loss_ids
        }

        count = jnp.asarray(optax.safe_increment(state.count))
        learning_rate = jnp.asarray(
            self.learning_rate(state.count)
            if isinstance(self.learning_rate, Schedule)
            else self.learning_rate
        )
        first_correction = 1.0 - self.b1**count
        second_correction = 1.0 - self.b2**count

        def parameter_update(first: JaxArray, largest_second: JaxArray) -> JaxArray:
            denominator = jnp.sqrt(largest_second / second_correction) + self.eps
            return -learning_rate * (first / first_correction) / denominator

        updates = tree.map(
            parameter_update,
            tree.map(lambda *m: reduce(jnp.add, m), *first_moments.values()),
            tree.map(lambda *v: reduce(jnp.maximum, v), *second_moments.values()),
        )
        new_state = MTAdamState[Parameters](count, first_moments, second_moments, norms)
        return updates, new_state
