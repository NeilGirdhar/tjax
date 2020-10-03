from typing import Any, Generic, List, Optional, Tuple, TypeVar

from ..annotations import PyTree
from ..dataclass import dataclass
from .transform import GradientTransformation, Weights

__all__ = ['ChainedGradientTransformation']


@dataclass
class ChainedGradientTransformation(GradientTransformation[List[PyTree], Weights],
                                    Generic[Weights]):

    U = TypeVar('U', bound='ChainedGradientTransformation[Weights]')

    transforms: List[GradientTransformation[Any, Weights]]

    def init(self, parameters: Weights) -> List[PyTree]:
        return [transform.init(parameters)
                for transform in self.transforms]

    def update(self,
               gradient: Weights,
               state: List[PyTree],
               parameters: Optional[Weights]) -> Tuple[Weights, List[PyTree]]:
        new_state: List[PyTree] = []
        for sub_state, transform in zip(state, self.transforms):
            gradient, new_sub_state = transform.update(gradient, sub_state, parameters)
            new_state.append(new_sub_state)
        return gradient, new_state
