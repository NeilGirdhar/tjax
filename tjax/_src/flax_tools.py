from __future__ import annotations

from dataclasses import field
from typing import Any

__all__ = ['flax_field']


def flax_field() -> Any:
    return field(init=False, default=None, kw_only=True)
