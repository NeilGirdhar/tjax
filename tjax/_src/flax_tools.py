from __future__ import annotations

from typing import Any

from tjax.dataclasses import field

__all__ = ['flax_field']


def flax_field() -> Any:
    """A field that contains submodules.

    This should be initialized in setup rather than being passed into the constructor.
    """
    return field(init=False, default=None, kw_only=True)  # pylint: disable=invalid-field-call
