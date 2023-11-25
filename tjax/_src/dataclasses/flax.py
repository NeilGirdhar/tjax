from __future__ import annotations

from typing import Any

from .helpers import field

__all__ = ['module_field']


def module_field(*, init: bool = False) -> Any:
    """A field that contains submodules."""
    return field(init=init, default=None, kw_only=True)  # pylint: disable=invalid-field-call
