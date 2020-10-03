from typing import Generic, Optional, Tuple, TypeVar

from ..annotations import PyTree
from ..dataclass import dataclass

__all__ = ['GradientTransformation', 'GradientState']


GradientState = PyTree
Weights = TypeVar('Weights', bound=PyTree)
State = TypeVar('State', bound=PyTree)


@dataclass
class GradientTransformation(Generic[State, Weights]):

    T = TypeVar('T', bound='GradientTransformation[State, Weights]')

    def init(self, parameters: Weights) -> State:
        raise NotImplementedError

    def update(self,
               gradient: Weights,
               state: State,
               parameters: Optional[Weights]) -> Tuple[Weights, State]:
        raise NotImplementedError
