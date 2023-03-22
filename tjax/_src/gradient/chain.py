from __future__ import annotations

from typing import Any, Generic

from typing_extensions import override

from ..annotations import PyTree
from ..dataclasses import dataclass
from .transform import GradientState, GradientTransformation, Weights

__all__ = ['ChainedGradientTransformation', 'ChainedGradientState']


@dataclass
class ChainedGradientState(GradientState):
    sub_states: list[PyTree]


@dataclass
class ChainedGradientTransformation(GradientTransformation[ChainedGradientState, Weights],
                                    Generic[Weights]):
    transforms: list[GradientTransformation[Any, Weights]]

    @override
    def init(self, parameters: Weights) -> ChainedGradientState:
        return ChainedGradientState([transform.init(parameters)
                                     for transform in self.transforms])

    @override
    def update(self,
               gradient: Weights,
               state: ChainedGradientState,
               parameters: Weights | None) -> tuple[Weights, ChainedGradientState]:
        new_state: list[PyTree] = []
        for sub_state, transform in zip(state.sub_states, self.transforms, strict=True):
            gradient, new_sub_state = transform.update(gradient, sub_state, parameters)
            new_state.append(new_sub_state)
        return gradient, ChainedGradientState(new_state)
