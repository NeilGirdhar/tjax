from __future__ import annotations

from typing import Any, Optional, Type

__all__ = ['dataclass_transform']


def dataclass_transform(cls: Optional[Type[Any]] = None, /, *, init: bool = True, repr_: bool =
                        True, eq: bool = True, order: bool = False) -> Any:
    return cls
