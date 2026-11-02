"""Transformations that combine or restrict other transformations."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import KW_ONLY
from typing import Any, cast, override

import optax
from optax import (
    GradientTransformation as OptaxGradientTransformation,
)

from tjax._src.annotations import (
    PyTree,
    RealNumeric,
)
from tjax.dataclasses import dataclass, field

from .transform import (
    GradientState,
    GradientTransformation,
    OptaxWrapper,
    accepted_extra_args,
    to_optax,
)


@dataclass
class ChainedGradientState(GradientState):
    """Gradient state for :class:`ChainedGradientTransformation`.

    Args:
        sub_states: The states of the transformations in the chain, in order.
    """

    sub_states: list[PyTree]


@dataclass
class ChainedGradientTransformation[Weights: PyTree](
    GradientTransformation[ChainedGradientState, Weights]
):
    """Apply a sequence of gradient transformations in order.

    Equivalent to optax's ``chain``, but expressed as a pytree-compatible
    dataclass so it can be passed through JIT boundaries and differentiated.

    Args:
        transforms: The ordered list of transformations to apply.  Each
            transformation sees the output of the previous one as its gradient.
    """

    transforms: list[GradientTransformation[Any, Weights]]

    @override
    def init(self, parameters: Weights) -> ChainedGradientState:
        return ChainedGradientState([transform.init(parameters) for transform in self.transforms])

    @override
    def update(
        self,
        gradient: Weights,
        state: ChainedGradientState,
        parameters: Weights | None,
        **extra_args: object,
    ) -> tuple[Weights, ChainedGradientState]:
        """Transform the weight gradient and update the gradient state.

        Args:
            gradient: The derivative of the loss with respect to the weights.
            state: The gradient state.
            parameters: The weights.
            extra_args: Keyword arguments, such as ``value``, that some transformations need.  Each
                transformation receives only those that its ``update`` declares.
        """
        new_state: list[PyTree] = []
        for sub_state, transform in zip(state.sub_states, self.transforms, strict=True):
            update = cast("Callable[..., tuple[Weights, PyTree]]", transform.update)
            gradient, new_sub_state = update(
                gradient, sub_state, parameters, **accepted_extra_args(update, extra_args)
            )
            new_state.append(new_sub_state)
        return gradient, ChainedGradientState(new_state)


@dataclass
class NamedChain[Weights: PyTree](OptaxWrapper[Weights]):
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
class Partition[Weights: PyTree](OptaxWrapper[Weights]):
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
class Masked[Weights: PyTree](OptaxWrapper[Weights]):
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
class Freeze[Weights: PyTree](OptaxWrapper[Weights]):
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
class SelectiveTransform[Weights: PyTree](OptaxWrapper[Weights]):
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


@dataclass
class Flatten[Weights: PyTree](OptaxWrapper[Weights]):
    """Flatten the parameters and gradients into one vector for the inner transformation.

    Args:
        inner: The transformation to apply to the flattened vector.
    """

    inner: GradientTransformation[Any, Weights]

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.flatten(to_optax(self.inner))


@dataclass
class Lookahead[Weights: PyTree](OptaxWrapper[Weights]):
    """Lookahead: a fast optimizer whose weights are periodically pulled toward slow weights.

    The parameters must be an ``optax.LookaheadParams`` holding the fast and slow weights, and
    the gradient must be with respect to the fast weights.

    References:
        Zhang et al., 2019,
            [Lookahead Optimizer: k steps forward, 1 step back](https://arxiv.org/abs/1907.08610)

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
