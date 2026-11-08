"""CAGrad, which improves the worst-off loss while staying near the average gradient."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, override

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from tjax._src.annotations import JaxRealArray, PyTree, RealNumeric
from tjax.dataclasses import dataclass, field

from .multi_loss import GradientCombiner

_STABILIZER = 1e-8
_NEWTON_STEPS = 60
_BARRIER_DECAY = 0.6
_LINE_SEARCH_STEPS = 24
_BOUNDARY_FRACTION = 0.99


def conflict_averse_weights(gram: JaxRealArray, radius: JaxRealArray) -> JaxRealArray:
    """Return the weights on the simplex that minimize CAGrad's objective.

    The objective is ``g_w . g_0 + radius ||g_w||``, where ``g_w`` is the weighted sum of the
    gradients and ``g_0`` is their mean.  It is convex, and has one variable for each loss.
    Gradients of very different sizes make it badly conditioned, which defeats first-order methods,
    so this follows the central path of a log barrier with Newton steps.

    Args:
        gram: The matrix of inner products of the gradients.
        radius: The coefficient of the norm term in the objective.
    """
    n_losses = gram.shape[0]
    mean_weights = jnp.full(n_losses, 1.0 / n_losses, gram.dtype)
    # Normalizing leaves the minimizer unchanged, and makes the barrier's schedule scale-free.
    scale = jnp.trace(gram) / n_losses
    scale = jnp.where(scale > 0.0, scale, 1.0)
    normalized = gram / scale
    normalized_radius = radius / jnp.sqrt(scale)
    stabilizer = _STABILIZER / scale
    line_search = jnp.asarray(0.5, gram.dtype) ** jnp.arange(_LINE_SEARCH_STEPS, dtype=gram.dtype)

    def barrier_objective(weights: JaxRealArray, barrier: JaxRealArray) -> JaxRealArray:
        norm = jnp.sqrt(weights @ normalized @ weights + stabilizer)
        linear = weights @ normalized @ mean_weights
        return linear + normalized_radius * norm - barrier * jnp.sum(jnp.log(weights))

    def newton_step(
        _: int, carry: tuple[JaxRealArray, JaxRealArray]
    ) -> tuple[JaxRealArray, JaxRealArray]:
        weights, barrier = carry
        gradient = jax.grad(barrier_objective)(weights, barrier)
        hessian = jax.hessian(barrier_objective)(weights, barrier)
        ones = jnp.ones((n_losses, 1), gram.dtype)
        kkt = jnp.block([[hessian, ones], [ones.T, jnp.zeros((1, 1), gram.dtype)]])
        rhs = jnp.concatenate([-gradient, jnp.zeros(1, gram.dtype)])
        direction = jnp.linalg.solve(kkt, rhs)[:n_losses]
        shrinking = direction < 0.0
        distance = -weights / jnp.where(shrinking, direction, -1.0)
        boundary = jnp.min(jnp.where(shrinking, distance, jnp.inf))
        steps = jnp.minimum(1.0, _BOUNDARY_FRACTION * boundary) * line_search
        values = jax.vmap(lambda step: barrier_objective(weights + step * direction, barrier))(
            steps
        )
        candidate = weights + steps[jnp.nanargmin(values)] * direction
        # Parallel gradients can make the Newton system singular, in which case keep the weights.
        new_weights = jnp.where(jnp.all(jnp.isfinite(candidate)), candidate, weights)
        return new_weights, _BARRIER_DECAY * barrier

    initial = (mean_weights, jnp.ones((), gram.dtype))
    return jax.lax.fori_loop(0, _NEWTON_STEPS, newton_step, initial)[0]


@dataclass
class CAGrad[Parameters: PyTree](GradientCombiner[None, Parameters]):
    """Move along the average gradient while improving the worst-off loss.

    Following the average gradient can make one loss much worse.  CAGrad instead looks within a ball
    around the average gradient for the direction that most improves whichever loss improves least.
    It therefore still converges on the average loss, and ``c`` trades that against the worst loss.

    References:
        Liu et al., 2021,
            [Conflict-Averse Gradient Descent for Multi-task
            Learning](https://arxiv.org/abs/2110.14048)

    Args:
        inner: The transformation applied to the combined gradient.
        c: The radius of the ball, as a fraction of the average gradient's norm, between 0 and 1.
            Zero gives the average gradient.
        rescale: How to shrink the direction: ``"quadratic"`` divides it by ``1 + c ** 2``, as the
            authors' code does by default, ``"linear"`` divides it by ``1 + c``, and ``"none"``
            leaves it unchanged, as in the paper's algorithm.
        loss_ids: The distinct IDs of the losses.
    """

    c: RealNumeric = 0.5
    rescale: Literal["none", "linear", "quadratic"] = field(default="quadratic", static=True)

    @override
    def combine(self, gradients: Sequence[Parameters], statistics: None) -> tuple[Parameters, None]:
        """Return the conflict-averse direction."""
        flattened = [ravel_pytree(gradient) for gradient in gradients]
        stacked = jnp.stack([flat for flat, _ in flattened])
        _, unravel = flattened[0]
        gram = stacked @ stacked.T
        mean_norm = jnp.sqrt(jnp.mean(gram) + _STABILIZER)
        radius = jnp.asarray(self.c * mean_norm + _STABILIZER)
        weighted = conflict_averse_weights(gram, radius) @ stacked
        multiplier = radius / (jnp.linalg.vector_norm(weighted) + _STABILIZER)
        direction = jnp.mean(stacked, axis=0) + multiplier * weighted
        match self.rescale:
            case "none":
                return unravel(direction), None
            case "linear":
                return unravel(direction / (1.0 + self.c)), None
            case "quadratic":
                return unravel(direction / (1.0 + jnp.square(self.c))), None
