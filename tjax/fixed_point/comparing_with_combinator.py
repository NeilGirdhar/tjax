from typing import Generic, TypeVar

from tjax import PyTree

from .augmented import State
from .combinator import IteratedFunctionWithCombinator
from .comparing import ComparingIteratedFunction, ComparingState
from .iterated_function import Parameters

__all__ = ['ComparingIteratedFunctionWithCombinator']


Comparand = TypeVar('Comparand', bound=PyTree)


class ComparingIteratedFunctionWithCombinator(
        IteratedFunctionWithCombinator[Parameters, State, ComparingState[State, Comparand]],
        ComparingIteratedFunction[Parameters, State, Comparand],
        Generic[Parameters, State, Comparand]):
    pass
