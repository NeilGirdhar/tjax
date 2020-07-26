from __future__ import annotations

from typing import Generic, TypeVar

from ..annotations import PyTree
from ..dataclass import dataclass

__all__ = ['AugmentedState']


State = TypeVar('State', bound=PyTree)


@dataclass
class AugmentedState(Generic[State]):

    current_state: State
    iterations: int
