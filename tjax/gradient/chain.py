from typing import Any, Callable, Generic, List, Optional, Tuple, TypeVar, Union

from chex import Numeric

from ..annotations import PyTree
from ..dataclass import dataclass
from .meta_parameter import MetaParameter
from .transform import GradientTransformation, Parameters

__all__ = ['ChainedGradientTransformation']


@dataclass
class ChainedGradientTransformation(GradientTransformation[List[PyTree], Parameters],
                                    Generic[Parameters]):

    U = TypeVar('U', bound='ChainedGradientTransformation[Parameters]')

    transforms: List[GradientTransformation[Any, Parameters]]

    def init(self, parameters: Parameters) -> List[PyTree]:
        return [transform.init(parameters)
                for transform in self.transforms]

    def update(self,
               gradient: Parameters,
               state: List[PyTree],
               parameters: Optional[Parameters]) -> Tuple[Parameters, List[PyTree]]:
        new_state: List[PyTree] = []
        for sub_state, transform in zip(state, self.transforms):
            gradient, new_state = transform.update(gradient, sub_state, parameters)
            new_state.append(new_state)
        return gradient, new_state

    def replace_meta_parameters(self: U,  # type: ignore
                                f: Callable[[MetaParameter], Union[Numeric, MetaParameter]]) -> U:
        new_transforms = []
        for transform in self.transforms:
            new_transforms.append(transform.replace_meta_parameters(f))
        return self.replace(transforms=new_transforms)
