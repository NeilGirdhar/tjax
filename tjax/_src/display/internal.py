from __future__ import annotations

from collections.abc import Callable
from typing import Any

from jax import tree
from rich.console import Console
from rich.tree import Tree

from .display_generic import display_generic

# Global variables ---------------------------------------------------------------------------------
global_console = Console()


# Functions ----------------------------------------------------------------------------------------
def internal_print_generic(*args: object,
                           raise_on_nan: bool = True,
                           as_leaves: bool = False,
                           console: Console | None = None,
                           ) -> None:
    if console is None:
        console = global_console
    found_nan = False
    root = Tree("", hide_root=True)
    for value in args:
        if as_leaves:
            value = node_to_leaves(value)  # noqa: PLW2901
        _ = root.add(display_generic(value, seen=set()))
        found_nan = found_nan or raise_on_nan and 'nan' in str(root)
    console.print(root)
    if found_nan:
        assert False  # noqa: PT015


def node_to_leaves(tree_: Any, is_leaf: Callable[[Any], bool] | None = None
                   ) -> list[tuple[str, Any]]:
    leaves_and_paths = tree.leaves_with_path(tree_, is_leaf=is_leaf)
    return [("".join(str(x) for x in key_path), leaf)
            for key_path, leaf in leaves_and_paths]
