"""Transformations that apply other transformations conditionally."""

from __future__ import annotations

from typing import Any, cast, override

import optax
from optax import (
    GradientTransformation as OptaxGradientTransformation,
)
from optax.transforms import ConditionFn

from tjax._src.annotations import (
    IntegralNumeric,
    PyTree,
)
from tjax.dataclasses import dataclass, field

from .transform import (
    GradientTransformation,
    OptaxWrapper,
    to_optax,
)


@dataclass
class ConditionallyMask[Weights: PyTree](OptaxWrapper[Weights]):
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
class ConditionallyTransform[Weights: PyTree](OptaxWrapper[Weights]):
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
class ApplyIfFinite[Weights: PyTree](OptaxWrapper[Weights]):
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
