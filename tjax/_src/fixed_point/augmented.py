from __future__ import annotations

from typing import Generic, TypeVar

from tjax.dataclasses import dataclass

from ..annotations import PyTree

__all__ = ['AugmentedState']


State = TypeVar('State', bound=PyTree)


@dataclass
class AugmentedState(Generic[State]):
    current_state: State
    iterations: int
