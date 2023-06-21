from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.tree import Tree

from .batch_dimensions import BatchDimensionIterator, BatchDimensions
from .display_generic import display_generic

__all__ = ['print_generic']


# Global variables ---------------------------------------------------------------------------------
global_console = Console()


# Functions ----------------------------------------------------------------------------------------
def print_generic(*args: Any,
                  batch_dims: BatchDimensions | None = None,
                  raise_on_nan: bool = True,
                  console: Console | None = None,
                  **kwargs: Any) -> None:
    if console is None:
        console = global_console
    bdi = BatchDimensionIterator(batch_dims)
    found_nan = False
    root = Tree("", hide_root=True)
    for value in args:
        sub_batch_dims = bdi.advance(value)
        root.add(display_generic(value, seen=set(), batch_dims=sub_batch_dims))
        found_nan = found_nan or raise_on_nan and 'nan' in str(root)
    for key, value in kwargs.items():
        sub_batch_dims = bdi.advance(value)
        root.add(display_generic(value, seen=set(), key=key, batch_dims=sub_batch_dims))
        found_nan = found_nan or raise_on_nan and 'nan' in str(root)
    console.print(root)
    if found_nan:
        assert False
