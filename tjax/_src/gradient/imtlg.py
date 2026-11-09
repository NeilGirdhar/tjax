"""IMTL-G, which weights gradients so that they make equal progress on each loss."""

from __future__ import annotations

from collections.abc import Sequence
from typing import override

import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from tjax._src.annotations import PyTree
from tjax._src.math_tools import divide_or_zero
from tjax.dataclasses import dataclass

from .multi_loss import GradientCombiner


@dataclass
class IMTLG[Parameters: PyTree](GradientCombiner[None, Parameters]):
    """Choose loss weights so that the combined gradient favours no loss.

    Weighted sums let a loss with a large gradient dominate.  IMTL-G instead chooses weights that
    sum to one so that the combined gradient has equal projections onto the direction of every
    loss's gradient.  This is the gradient-balancing half of IMTL.

    Pass the gradients with respect to the parameters that the losses share.

    References:
        Liu et al., 2021,
            [Towards Impartial Multi-task Learning](https://openreview.net/forum?id=IMPnRXEWpvr)

    Args:
        inner: The transformation applied to the combined gradient.
        loss_ids: The distinct IDs of the losses.
    """

    @override
    def init_statistics(self, parameters: Parameters) -> None:
        return None

    @override
    def combine(self, gradients: Sequence[Parameters], statistics: None) -> tuple[Parameters, None]:
        """Return the combined gradient.

        The weights have a closed form, which is a linear system with one row for each loss after
        the first.  Parallel gradients make that system singular, so a least-squares solve picks
        the smallest weights that satisfy it.
        """
        if len(gradients) == 1:
            return gradients[0], None
        flattened = [ravel_pytree(gradient) for gradient in gradients]
        stacked = jnp.stack([flat for flat, _ in flattened])
        _, unravel = flattened[0]
        norms = jnp.linalg.vector_norm(stacked, axis=1, keepdims=True)
        directions = divide_or_zero(stacked, norms)
        differences = stacked[:1] - stacked[1:]
        direction_differences = directions[:1] - directions[1:]
        tail, *_ = jnp.linalg.lstsq(
            direction_differences @ differences.T, direction_differences @ stacked[0]
        )
        weights = jnp.concatenate([1.0 - jnp.sum(tail, keepdims=True), tail])
        return unravel(weights @ stacked), None
