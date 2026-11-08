"""PCGrad, which removes conflicts between the gradients of several losses."""

from __future__ import annotations

from collections.abc import Sequence
from typing import override

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.flatten_util import ravel_pytree

from tjax._src.annotations import JaxArray, KeyArray, PyTree
from tjax._src.math_tools import divide_or_zero
from tjax.dataclasses import dataclass

from .multi_loss import GradientCombiner


@dataclass
class PCGrad[Parameters: PyTree](GradientCombiner[KeyArray, Parameters]):
    """Project away the components of each loss's gradient that conflict with the others.

    When two gradients point in opposing directions, one loss's progress undoes the other's, and
    a sum can favour whichever gradient is larger.  PCGrad instead removes from each gradient its
    component along any gradient that it conflicts with, and sums the results.  The statistics are
    the random key that sets the order of the projections.

    References:
        Yu et al., 2020,
            [Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782)

    Args:
        inner: The transformation applied to the combined gradient.
        key: The key for choosing the order of the projections.
        loss_ids: The distinct IDs of the losses.
    """

    key: KeyArray

    @override
    def init_statistics(self, parameters: Parameters) -> KeyArray:
        return self.key

    @override
    def combine(
        self, gradients: Sequence[Parameters], statistics: KeyArray
    ) -> tuple[Parameters, KeyArray]:
        """Return the sum of the projected gradients and the next key.

        Each gradient is projected against the original gradients of the other losses, in an order
        that is random and separate for each loss, since the result depends on the order.
        """
        flattened = [ravel_pytree(gradient) for gradient in gradients]
        stacked = jnp.stack([flat for flat, _ in flattened])
        _, unravel = flattened[0]
        n_losses = len(gradients)
        next_key, subkey = jr.split(statistics)

        def project(index: JaxArray, order_key: KeyArray) -> JaxArray:
            def project_onto(gradient: JaxArray, other_index: JaxArray) -> tuple[JaxArray, None]:
                other = stacked[other_index]
                dot = jnp.vecdot(gradient, other)
                coefficient = divide_or_zero(dot, jnp.vecdot(other, other))
                conflicts = (other_index != index) & (dot < 0.0)
                return jnp.where(conflicts, gradient - coefficient * other, gradient), None

            order = jr.permutation(order_key, n_losses)
            return jax.lax.scan(project_onto, stacked[index], order)[0]

        projected = jax.vmap(project)(jnp.arange(n_losses), jr.split(subkey, n_losses))
        return unravel(jnp.sum(projected, axis=0)), next_key
