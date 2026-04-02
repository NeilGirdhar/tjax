from __future__ import annotations

from typing import Any, override

from tjax._src.annotations import PyTree
from tjax.dataclasses import dataclass

from .transform import GradientState, GradientTransformation


@dataclass
class ChainedGradientState(GradientState):
    """Gradient state for :class:`ChainedGradientTransformation`.

    Stores the individual states of each transformation in the chain.
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
        self, gradient: Weights, state: ChainedGradientState, parameters: Weights | None
    ) -> tuple[Weights, ChainedGradientState]:
        new_state: list[PyTree] = []
        for sub_state, transform in zip(state.sub_states, self.transforms, strict=True):
            gradient, new_sub_state = transform.update(gradient, sub_state, parameters)
            new_state.append(new_sub_state)
        return gradient, ChainedGradientState(new_state)
