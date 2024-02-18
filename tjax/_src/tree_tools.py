from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

__all__ = ['tree_map_with_path']


T = TypeVar('T')


def tree_map_with_path(structure: Any,
                       transform: Callable[[T, tuple[str, ...]], T]
                       ) -> Any:
    def f(structure: Any,
          transform: Callable[[T, tuple[str, ...]], T],
          path: tuple[str, ...]
          ) -> Any:
        if isinstance(structure, dict):
            out_structure = {}
            for key, value in structure.items():
                out_structure[key] = f(value, transform, (*path, key))
            return out_structure
        return transform(structure, path)
    return f(structure, transform, ())
