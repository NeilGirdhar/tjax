from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.tree import Tree

from .display_generic import display_generic

# Global variables ---------------------------------------------------------------------------------
global_console = Console()


# Functions ----------------------------------------------------------------------------------------
def internal_print_generic(*args: Any,
                           raise_on_nan: bool = True,
                           console: Console | None = None,
                           **kwargs: Any) -> None:
    if console is None:
        console = global_console
    found_nan = False
    root = Tree("", hide_root=True)
    for value in args:
        _ = root.add(display_generic(value, seen=set()))
        found_nan = found_nan or raise_on_nan and 'nan' in str(root)
    for key, value in kwargs.items():
        _ = root.add(display_generic(value, seen=set(), key=key))
        found_nan = found_nan or raise_on_nan and 'nan' in str(root)
    console.print(root)
    if found_nan:
        assert False  # noqa: PT015
