from __future__ import annotations

from typing import Any, Generic, List, Optional, Tuple

from ..annotations import PyTree
from ..dataclasses import dataclass
from .transform import GradientState, GradientTransformation, Weights

__all__ = ['ChainedGradientTransformation']


@dataclass
class ChainedGradientState(GradientState):
    sub_states: List[PyTree]


@dataclass
class ChainedGradientTransformation(GradientTransformation[ChainedGradientState, Weights],
                                    Generic[Weights]):
    transforms: List[GradientTransformation[Any, Weights]]

    def init(self, parameters: Weights) -> ChainedGradientState:
        return ChainedGradientState([transform.init(parameters)
                                     for transform in self.transforms])

    def update(self,
               gradient: Weights,
               state: ChainedGradientState,
               parameters: Optional[Weights]) -> Tuple[Weights, ChainedGradientState]:
        new_state: List[PyTree] = []
        for sub_state, transform in zip(state.sub_states, self.transforms):
            gradient, new_sub_state = transform.update(gradient, sub_state, parameters)
            new_state.append(new_sub_state)
        return gradient, ChainedGradientState(new_state)
