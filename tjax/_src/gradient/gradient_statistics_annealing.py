"""Gradient-statistics annealing, which weights losses against a reference gradient."""

from __future__ import annotations

from collections.abc import Sequence
from typing import override

import jax.numpy as jnp
from jax import tree
from jax.flatten_util import ravel_pytree

from tjax._src.annotations import JaxRealArray, PyTree, RealNumeric
from tjax.dataclasses import dataclass

from .multi_loss import GradientCombiner


@dataclass
class GradientStatisticsAnnealing[Parameters: PyTree](GradientCombiner[JaxRealArray, Parameters]):
    """Weight each loss so that its gradient is comparable to a reference loss's gradient.

    When one loss's gradient dwarfs the others, as with a physics-informed network's residual and
    its boundary conditions, the others barely train.  This method keeps a running weight for each
    loss, and sums the weighted gradients.  The first loss ID names the reference loss, whose weight
    stays one.  The statistics are the other losses' weights.

    References:
        Wang et al., 2020,
            [Understanding and mitigating gradient pathologies in physics-informed neural
            networks](https://arxiv.org/abs/2001.04536)

    Args:
        inner: The transformation applied to the combined gradient.
        alpha: The weight of each new estimate in the running weights.
    """

    alpha: RealNumeric = 0.9

    @override
    def init_statistics(self, parameters: Parameters, loss_ids: tuple[str, ...]) -> JaxRealArray:
        return jnp.ones(len(loss_ids) - 1)

    @override
    def combine(
        self, gradients: Sequence[Parameters], statistics: JaxRealArray
    ) -> tuple[Parameters, JaxRealArray]:
        """Return the weighted sum of the gradients and the next weights.

        Each loss's new estimate is the largest gradient magnitude of the reference loss divided by
        the mean gradient magnitude of that loss.  A loss with a zero gradient keeps its weight.
        """
        reference, *others = gradients
        if not others:
            return reference, statistics
        peak = jnp.max(jnp.abs(ravel_pytree(reference)[0]))
        means = jnp.stack([jnp.mean(jnp.abs(ravel_pytree(g)[0])) for g in others])
        estimates = peak / jnp.where(means > 0.0, means, 1.0)
        weights = jnp.where(
            means > 0.0, (1.0 - self.alpha) * statistics + self.alpha * estimates, statistics
        )
        combined = tree.map(
            lambda r, *o: r + sum(w * g for w, g in zip(weights, o, strict=True)),
            reference,
            *others,
        )
        return combined, weights
