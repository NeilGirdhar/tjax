"""GradVac, which steers pairs of gradients toward the similarity they usually have."""

from __future__ import annotations

from collections.abc import Sequence
from typing import override

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import tree

from tjax._src.annotations import JaxArray, JaxRealArray, KeyArray, PyTree, RealNumeric
from tjax._src.math_tools import divide_or_zero
from tjax.dataclasses import dataclass

from .multi_loss import GradientCombiner


@dataclass
class GradVacStatistics:
    """Statistics for :class:`GradVac`.

    Args:
        key: The key for choosing the order of the adjustments.
        similarities: A tree with the structure of the parameters whose leaves are matrices.  Each
            entry is the running cosine similarity of a pair of losses on that parameter block.
    """

    key: KeyArray
    similarities: PyTree


@dataclass
class GradVac[Parameters: PyTree](GradientCombiner[GradVacStatistics, Parameters]):
    """Steer each pair of gradients toward the similarity that the pair usually has.

    PCGrad acts only on conflicting gradients, and removes the conflict entirely.  But related
    losses should have positively aligned gradients, and how much depends on the pair, the layer,
    and the stage of training.  GradVac keeps a running similarity for each pair of losses on each
    parameter block, which is a leaf of the parameters.  It adjusts a gradient whenever its
    similarity falls below that target.  With a target of zero, this is PCGrad's projection.

    References:
        Wang et al., 2020,
            [Gradient Vaccine: Investigating and Improving Multi-task Optimization in Massively
            Multilingual Models](https://arxiv.org/abs/2010.05874)

    Args:
        inner: The transformation applied to the combined gradient.
        key: The key for choosing the order of the adjustments.
        beta: The weight of each new similarity in the running similarities.
        loss_ids: The distinct IDs of the losses.
    """

    key: KeyArray
    beta: RealNumeric = 1e-2

    @override
    def init_statistics(self, parameters: Parameters) -> GradVacStatistics:
        n_losses = len(self.loss_ids)
        similarities = tree.map(lambda x: jnp.zeros((n_losses, n_losses), x.dtype), parameters)
        return GradVacStatistics(self.key, similarities)

    def _combine_block(
        self, stacked: JaxRealArray, similarities: JaxRealArray, key: KeyArray
    ) -> tuple[JaxRealArray, JaxRealArray]:
        """Return the sum of one block's adjusted gradients and its next similarities.

        Each gradient is adjusted against the original gradients of the other losses, in a random
        order that is separate for each loss.  The adjustment adds the multiple of the other
        gradient that brings their similarity up to the target.

        Args:
            stacked: The block's gradients, with one row for each loss.
            similarities: The block's running similarities.
            key: The key for choosing the order of the adjustments.
        """
        n_losses = stacked.shape[0]

        def adjust(
            index: JaxArray, targets: JaxRealArray, order_key: KeyArray
        ) -> tuple[JaxRealArray, JaxRealArray]:
            def adjust_toward(
                carry: tuple[JaxRealArray, JaxRealArray], other_index: JaxArray
            ) -> tuple[tuple[JaxRealArray, JaxRealArray], None]:
                gradient, targets = carry
                other = stacked[other_index]
                gradient_norm = jnp.linalg.vector_norm(gradient)
                other_norm = jnp.linalg.vector_norm(other)
                similarity = divide_or_zero(jnp.vecdot(gradient, other), gradient_norm * other_norm)
                target = targets[other_index]
                target_sine = jnp.sqrt(jnp.maximum(1.0 - jnp.square(target), 0.0))
                sine = jnp.sqrt(jnp.maximum(1.0 - jnp.square(similarity), 0.0))
                coefficient = divide_or_zero(
                    gradient_norm * (target * sine - similarity * target_sine),
                    other_norm * target_sine,
                )
                is_other = other_index != index
                needs_adjustment = is_other & (similarity < target)
                adjusted = jnp.where(needs_adjustment, gradient + coefficient * other, gradient)
                new_target = (1.0 - self.beta) * target + self.beta * similarity
                new_targets = targets.at[other_index].set(jnp.where(is_other, new_target, target))
                return (adjusted, new_targets), None

            order = jr.permutation(order_key, n_losses)
            return jax.lax.scan(adjust_toward, (stacked[index], targets), order)[0]

        adjusted, new_similarities = jax.vmap(adjust)(
            jnp.arange(n_losses), similarities, jr.split(key, n_losses)
        )
        return jnp.sum(adjusted, axis=0), new_similarities

    @override
    def combine(
        self, gradients: Sequence[Parameters], statistics: GradVacStatistics
    ) -> tuple[Parameters, GradVacStatistics]:
        """Return the sum of the adjusted gradients and the next statistics."""
        treedef = tree.structure(gradients[0])
        blocks = zip(*(tree.leaves(gradient) for gradient in gradients), strict=True)
        block_similarities = tree.leaves(statistics.similarities)
        next_key, subkey = jr.split(statistics.key)
        block_keys = jr.split(subkey, len(block_similarities))
        results = [
            self._combine_block(
                jnp.stack([jnp.ravel(leaf) for leaf in block]), similarities, block_key
            )
            for block, similarities, block_key in zip(
                blocks, block_similarities, block_keys, strict=True
            )
        ]
        shapes = [leaf.shape for leaf in tree.leaves(gradients[0])]
        combined = tree.unflatten(
            treedef,
            [jnp.reshape(total, shape) for (total, _), shape in zip(results, shapes, strict=True)],
        )
        similarities = tree.unflatten(treedef, [similarity for _, similarity in results])
        return combined, GradVacStatistics(next_key, similarities)
