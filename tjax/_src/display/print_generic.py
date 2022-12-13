from __future__ import annotations

from typing import Any, Optional, Tuple

from .batch_dimensions import BatchDimensionIterator
from .display_generic import display_generic, display_key_and_value

__all__ = ['print_generic']


def print_generic(*args: Any,
                  batch_dims: Optional[Tuple[Optional[int], ...]] = None,
                  raise_on_nan: bool = True,
                  **kwargs: Any) -> None:
    bdi = BatchDimensionIterator(batch_dims)
    found_nan = False
    for value in args:
        sub_batch_dims = bdi.advance(value)
        s = display_generic(value, set(), batch_dims=sub_batch_dims)
        print(s)
        found_nan = found_nan or raise_on_nan and 'nan' in str(s)
    for key, value in kwargs.items():
        sub_batch_dims = bdi.advance(value)
        s = display_key_and_value(key, value, "=", set(), batch_dims=sub_batch_dims)
        print(s)
        found_nan = found_nan or raise_on_nan and 'nan' in str(s)
    if found_nan:
        assert False
