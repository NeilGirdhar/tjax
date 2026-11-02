"""Gradient transformations that wrap other gradient transformations.

Each combinator holds its inner transformations as pytree fields, so it can pass through jit like
any other tjax transformation, and delegates to the optax combinator of the same name.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import KW_ONLY
from typing import Any, cast, override

import jax.numpy as jnp
import optax
from optax import (
    GradientTransformation as OptaxGradientTransformation,
    GradientTransformationExtraArgs,
)
from optax.transforms import ConditionFn, ShouldSkipUpdateFunction

from tjax._src.annotations import IntegralNumeric, PyTree, RealNumeric
from tjax.dataclasses import dataclass, field

from .transform import GenericGradientState, GradientTransformation, accepted_extra_args

type _Update = Callable[..., tuple[Any, PyTree]]


def to_optax(transform: GradientTransformation[Any, Any]) -> GradientTransformationExtraArgs:
    """Convert a tjax gradient transformation into an optax one.

    The optax update forwards to ``transform.update`` only the extra keyword arguments that it
    declares.

    Args:
        transform: The tjax transformation to convert.
    """

    def init_fn(params: PyTree) -> PyTree:
        return transform.init(params)

    def update_fn(
        updates: PyTree, state: PyTree, params: PyTree = None, **extra_args: object
    ) -> tuple[PyTree, PyTree]:
        update = cast("_Update", transform.update)
        return update(updates, state, params, **accepted_extra_args(update, extra_args))

    return GradientTransformationExtraArgs(init_fn, update_fn)


@dataclass
class _OptaxCombinator[Weights: PyTree](GradientTransformation[GenericGradientState, Weights]):
    """A transformation that delegates to the optax transformation returned by ``_optax``."""

    def _optax(self) -> OptaxGradientTransformation:
        raise NotImplementedError

    @override
    def init(self, parameters: Weights) -> GenericGradientState:
        return GenericGradientState(self._optax().init(parameters))

    @override
    def update(
        self,
        gradient: Weights,
        state: GenericGradientState,
        parameters: Weights | None,
        **extra_args: object,
    ) -> tuple[Weights, GenericGradientState]:
        """Transform the weight gradient and update the gradient state.

        Args:
            gradient: The derivative of the loss with respect to the weights.
            state: The gradient state.
            parameters: The weights.
            extra_args: Keyword arguments, such as ``value``, that inner transformations need.
                Each transformation receives only those that its ``update`` declares.
        """
        update = cast("_Update", self._optax().update)
        new_gradient, new_state = update(
            gradient, state.data, parameters, **accepted_extra_args(update, extra_args)
        )
        return new_gradient, GenericGradientState(new_state)


# Masking and partitioning -------------------------------------------------------------------------
@dataclass
class Masked[Weights: PyTree](_OptaxCombinator[Weights]):
    """Apply a transformation to some of the updates and pass the rest through unchanged.

    Args:
        inner: The transformation to apply to the selected updates.
        mask: A tree with the same structure as (or a prefix of) the parameters, or a function of
            the parameters returning one.  Its boolean leaves are true for the updates to
            transform.
        mask_compatible_extra_args: Whether to also mask extra arguments whose tree structure
            matches the parameters.
    """

    inner: GradientTransformation[Any, Weights]
    mask: PyTree | Callable[[Weights], PyTree] = field(static=True)
    _: KW_ONLY
    mask_compatible_extra_args: bool = field(default=False, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.masked(
            to_optax(self.inner),
            self.mask,
            mask_compatible_extra_args=self.mask_compatible_extra_args,
        )


@dataclass
class Partition[Weights: PyTree](_OptaxCombinator[Weights]):
    """Partition the parameters and apply a different transformation to each subset.

    Args:
        transforms: The transformation to apply to each label.
        param_labels: A tree with the same structure as (or a prefix of) the parameters, or a
            function of the parameters returning one.  Its leaves are keys of ``transforms``.
        mask_compatible_extra_args: Whether to also partition extra arguments whose tree
            structure matches the parameters.
    """

    transforms: Mapping[Hashable, GradientTransformation[Any, Weights]]
    param_labels: PyTree | Callable[[Weights], PyTree] = field(static=True)
    _: KW_ONLY
    mask_compatible_extra_args: bool = field(default=False, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.partition(
            {label: to_optax(transform) for label, transform in self.transforms.items()},
            self.param_labels,
            mask_compatible_extra_args=self.mask_compatible_extra_args,
        )


MultiTransform = Partition


@dataclass
class Freeze[Weights: PyTree](_OptaxCombinator[Weights]):
    """Zero the updates of frozen parameters.

    Args:
        mask: A boolean, or a tree with the same structure as (or a prefix of) the parameters,
            that is true for the parameters to freeze.
    """

    mask: PyTree = field(static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.freeze(self.mask)


@dataclass
class SelectiveTransform[Weights: PyTree](_OptaxCombinator[Weights]):
    """Apply a transformation only to the parameters that are not frozen.

    Frozen parameters receive zero updates and no optimizer state.

    Args:
        optimizer: The transformation to apply to the unfrozen parameters.
        freeze_mask: A boolean, or a tree with the same structure as (or a prefix of) the
            parameters, that is true for the parameters to freeze.
    """

    optimizer: GradientTransformation[Any, Weights]
    _: KW_ONLY
    freeze_mask: PyTree = field(static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.selective_transform(to_optax(self.optimizer), freeze_mask=self.freeze_mask)


# Chaining and layout ------------------------------------------------------------------------------
@dataclass
class NamedChain[Weights: PyTree](_OptaxCombinator[Weights]):
    """Apply a sequence of named transformations in order.

    The state is a dictionary from each name to its transformation's state.

    Args:
        names: The name of each transformation.
        transforms: The transformations, in the order to apply them.
    """

    names: Sequence[str] = field(static=True)
    transforms: Sequence[GradientTransformation[Any, Weights]]

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.named_chain(
            *(
                (name, to_optax(transform))
                for name, transform in zip(self.names, self.transforms, strict=True)
            )
        )


@dataclass
class Flatten[Weights: PyTree](_OptaxCombinator[Weights]):
    """Flatten the parameters and gradients into one vector for the inner transformation.

    Args:
        inner: The transformation to apply to the flattened vector.
    """

    inner: GradientTransformation[Any, Weights]

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.flatten(to_optax(self.inner))


@dataclass
class Stateless[Weights: PyTree](_OptaxCombinator[Weights]):
    """Create a stateless transformation from a function of the updates and parameters.

    Args:
        f: A function from the updates and parameters (or ``None``) to new updates.
    """

    f: Callable[[PyTree, PyTree], PyTree] = field(static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.stateless(self.f)


@dataclass
class StatelessWithTreeMap[Weights: PyTree](_OptaxCombinator[Weights]):
    """Create a stateless transformation that maps a function over the update leaves.

    Args:
        f: A function from an update leaf and its parameter leaf (or ``None``) to a new update
            leaf.
    """

    f: Callable[[Any, Any], Any] = field(static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.stateless_with_tree_map(self.f)


# Conditional application --------------------------------------------------------------------------
@dataclass
class ConditionallyMask[Weights: PyTree](_OptaxCombinator[Weights]):
    """Apply a transformation only at some steps, and emit zero updates at the others.

    Args:
        inner: The transformation to apply.
        should_transform_fn: A function from the step count to whether to apply ``inner``.
        forward_extra_args: Whether to pass the step count's extra arguments on to ``inner``.
    """

    inner: GradientTransformation[Any, Weights]
    should_transform_fn: ConditionFn = field(static=True)
    forward_extra_args: bool = field(default=False, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.conditionally_mask(
            to_optax(self.inner), self.should_transform_fn, self.forward_extra_args
        )


@dataclass
class ConditionallyTransform[Weights: PyTree](_OptaxCombinator[Weights]):
    """Apply a transformation only at some steps, and pass the updates through at the others.

    Args:
        inner: The transformation to apply.
        should_transform_fn: A function from the step count to whether to apply ``inner``.
        forward_extra_args: Whether to pass the step count's extra arguments on to ``inner``.
    """

    inner: GradientTransformation[Any, Weights]
    should_transform_fn: ConditionFn = field(static=True)
    forward_extra_args: bool = field(default=False, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.conditionally_transform(
            to_optax(self.inner), self.should_transform_fn, self.forward_extra_args
        )


@dataclass
class ApplyIfFinite[Weights: PyTree](_OptaxCombinator[Weights]):
    """Skip updates whose gradients contain NaNs or infinities.

    A skipped step emits zero updates and leaves the inner state unchanged.  After
    ``max_consecutive_errors`` consecutive skips, the updates are applied anyway.

    Args:
        inner: The transformation to protect.
        max_consecutive_errors: The number of consecutive non-finite steps to skip.
    """

    inner: GradientTransformation[Any, Weights]
    max_consecutive_errors: IntegralNumeric

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.apply_if_finite(
            to_optax(self.inner),
            cast("int", self.max_consecutive_errors),
        )


# Accumulation -------------------------------------------------------------------------------------
@dataclass
class MultiSteps[Weights: PyTree](_OptaxCombinator[Weights]):
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

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.MultiSteps(
            to_optax(self.opt),
            self.every_k_schedule,
            use_grad_mean=self.use_grad_mean,
            should_skip_update_fn=self.should_skip_update_fn,
            accumulator_dtype=self.accumulator_dtype,
        ).gradient_transformation()


@dataclass
class Lookahead[Weights: PyTree](_OptaxCombinator[Weights]):
    """Lookahead: a fast optimizer whose weights are periodically pulled toward slow weights.

    The parameters must be an ``optax.LookaheadParams`` holding the fast and slow weights, and
    the gradient must be with respect to the fast weights.

    References:
        [Zhang et al, 2019](https://arxiv.org/abs/1907.08610)

    Args:
        fast_optimizer: The transformation that updates the fast weights.
        sync_period: The number of fast steps between synchronizations.
        slow_step_size: The step size for moving the slow weights toward the fast weights.
        reset_state: Whether to reset the fast optimizer's state after each synchronization.
    """

    fast_optimizer: GradientTransformation[Any, Weights]
    sync_period: int = field(static=True)
    slow_step_size: RealNumeric
    reset_state: bool = field(default=False, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.lookahead(
            to_optax(self.fast_optimizer),
            self.sync_period,
            self.slow_step_size,
            reset_state=self.reset_state,
        )


# Monitoring ---------------------------------------------------------------------------------------
@dataclass
class Snapshot[Weights: PyTree](_OptaxCombinator[Weights]):
    """Store a measurement of the updates in the state, and pass the updates through.

    Args:
        measure_name: The key under which the state stores the measurement.
        measure: A function from the updates to the measurement.
    """

    measure_name: str = field(static=True)
    measure: Callable[[PyTree], PyTree] = field(static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.snapshot(self.measure_name, self.measure)


@dataclass
class MeasureWithEma[Weights: PyTree](_OptaxCombinator[Weights]):
    """Store an exponential moving average of a measurement of the updates in the state.

    The updates pass through unchanged.

    Args:
        measure: A function from the updates to the measurement.
        decay: The decay rate of the moving average.
        debias: Whether to debias the moving average.
        accumulator_dtype: Optional dtype for the moving average.
    """

    measure: Callable[[PyTree], PyTree] = field(static=True)
    decay: RealNumeric
    debias: bool = field(default=True, static=True)
    accumulator_dtype: Any = field(default=None, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.measure_with_ema(
            self.measure, self.decay, debias=self.debias, accumulator_dtype=self.accumulator_dtype
        )
