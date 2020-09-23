from typing import Generic, TypeVar

from tjax import PyTree

from .augmented import State
from .combinator import Differentiand, IteratedFunctionWithCombinator
from .comparing import ComparingIteratedFunction, ComparingState
from .iterated_function import Parameters, Trajectory

__all__ = ['ComparingIteratedFunctionWithCombinator']


Comparand = TypeVar('Comparand', bound=PyTree)


class ComparingIteratedFunctionWithCombinator(
        IteratedFunctionWithCombinator[Parameters, State, Comparand, Differentiand, Trajectory,
                                       ComparingState[State, Comparand]],
        ComparingIteratedFunction[Parameters, State, Comparand, Trajectory],
        Generic[Parameters, State, Comparand, Differentiand, Trajectory]):
    pass
