from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any, cast, override

from tjax._src.annotations import PyTree
from tjax.dataclasses import dataclass

from .transform import GradientState, GradientTransformation


@dataclass
class ChainedGradientState(GradientState):
    """Gradient state for :class:`ChainedGradientTransformation`.

    Stores the individual states of each transformation in the chain.
    """

    sub_states: list[PyTree]


def _accepted_extra_args(
    update: Callable[..., object], extra_args: Mapping[str, object]
) -> dict[str, object]:
    """Return the subset of extra_args that update accepts as keyword arguments."""
    parameters = inspect.signature(update).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return dict(extra_args)
    return {
        name: value
        for name, value in extra_args.items()
        if name in parameters and parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    }


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
                gradient, sub_state, parameters, **_accepted_extra_args(update, extra_args)
            )
            new_state.append(new_sub_state)
        return gradient, ChainedGradientState(new_state)
