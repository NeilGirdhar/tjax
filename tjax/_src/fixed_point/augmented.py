from __future__ import annotations

from typing import Generic, TypeVar

from tjax.dataclasses import dataclass

from ..annotations import JaxIntegralArray, PyTree

State = TypeVar('State', bound=PyTree)


@dataclass
class AugmentedState(Generic[State]):
    current_state: State
    iterations: JaxIntegralArray
