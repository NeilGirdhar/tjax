"""Transformations that accumulate updates over steps."""

from __future__ import annotations

from typing import Any, override

import jax.numpy as jnp
import optax
from optax import (
    GradientTransformation as OptaxGradientTransformation,
)
from optax.transforms import ShouldSkipUpdateFunction

from tjax._src.annotations import JaxBooleanArray, PyTree, RealNumeric
from tjax.dataclasses import as_shallow_dict, dataclass, field

from .transform import GenericGradientState, GradientTransformation, OptaxWrapper, to_optax


@dataclass
class Trace[Weights: PyTree](OptaxWrapper[Weights]):
    """Compute a trace of past updates.

    Note: `trace` and `ema` have very similar but distinct updates;
    `trace = decay * trace + t`, while `ema = decay * ema + (1-decay) * t`.
    Both are frequently found in the optimisation literature.

    Args:
        decay: the decay rate for the trace of past updates.
        nesterov: whether to use Nesterov momentum.
        accumulator_dtype: optional `dtype` to be used for the accumulator; if
            `None` then the `dtype` is inferred from `params` and `updates`.
    """

    decay: RealNumeric
    nesterov: bool = field(default=False, static=True)
    accumulator_dtype: Any = field(default=None, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.trace(**as_shallow_dict(self))


@dataclass
class EMA[Weights: PyTree](OptaxWrapper[Weights]):
    """Compute an exponential moving average of past updates.

    Note: `trace` and `ema` have very similar but distinct updates;
    `ema = decay * ema + (1-decay) * t`, while `trace = decay * trace + t`.
    Both are frequently found in the optimisation literature.

    Args:
        decay: the decay rate for the exponential moving average.
        debias: whether to debias the transformed gradient.
        accumulator_dtype: optional `dtype` to be used for the accumulator; if `None`
            then the `dtype` is inferred from `params` and `updates`.
    """

    decay: RealNumeric
    debias: bool = field(default=True, static=True)
    accumulator_dtype: Any = field(default=None, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.ema(**as_shallow_dict(self))


@dataclass
class ApplyEvery[Weights: PyTree](OptaxWrapper[Weights]):
    """Accumulate gradients and apply them every k steps.

    Note that if this transformation is part of a chain, the states of the other
    transformations will still be updated at every step. In particular, using
    `apply_every` with a batch size of N/2 and k=2 is not necessarily equivalent
    to not using `apply_every` with a batch size of N. If this equivalence is
    important for you, consider using the `optax.MultiSteps`.

    Args:
        k: emit non-zero gradients every k steps, otherwise accumulate them.
    """

    k: int = field(default=1, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.apply_every(**as_shallow_dict(self))


@dataclass
class MultiSteps[Weights: PyTree](OptaxWrapper[Weights]):
    """Accumulate gradients over several mini-steps and apply the inner transformation once.

    Between applications, the updates are zero.

    Args:
        opt: The transformation to apply to the accumulated gradients.
        every_k_schedule: The number of mini-steps per update, or a function from the update
            count to that number.
        use_grad_mean: Whether to average the accumulated gradients instead of summing them.
        should_skip_update_fn: An optional function of the updates, gradient step and
            parameters that decides whether to skip a mini-step.
        accumulator_dtype: The dtype of the gradient accumulator.
    """

    opt: GradientTransformation[Any, Weights]
    every_k_schedule: int | optax.Schedule = field(static=True)
    use_grad_mean: bool = field(default=True, static=True)
    should_skip_update_fn: ShouldSkipUpdateFunction | None = field(default=None, static=True)
    accumulator_dtype: Any = field(default=jnp.float32, static=True)

    def _multi_steps(self) -> optax.MultiSteps:
        return optax.MultiSteps(
            to_optax(self.opt),
            self.every_k_schedule,
            use_grad_mean=self.use_grad_mean,
            should_skip_update_fn=self.should_skip_update_fn,
            accumulator_dtype=self.accumulator_dtype,
        )

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return self._multi_steps().gradient_transformation()

    def has_updated(self, state: GenericGradientState) -> JaxBooleanArray:
        """Return whether the update that produced this state applied the inner transformation.

        Between applications the updates are zero, so this tells a training loop when the
        parameters actually changed.

        Args:
            state: The gradient state returned by ``update``.
        """
        return jnp.asarray(self._multi_steps().has_updated(state.data))
