from __future__ import annotations

from typing import Any

from jax import debug, tree
from rich.console import Console

from .internal import internal_print_generic


def print_generic(*args: Any,
                  raise_on_nan: bool = True,
                  immediate: bool = False,
                  console: Console | None = None,
                  **kwargs: Any
                  ) -> None:
    """Uses internal_print_generic in a tapped function.

    Args:
        raise_on_nan: Assert if NaN is found in the output.
        immediate: Print the value immediately even if they're tracers.  Do not wait for the
            compiled function to be called.
        console: The console that formats the output.
        args: Positional arguments to be printed.  Only dynamic arguments are allowed.
        kwargs: Keyword arguments to be printed.  Only static keys and dynamic values are allowed.
    """
    if immediate:
        internal_print_generic(*args, raise_on_nan=raise_on_nan, console=console, **kwargs)
        return

    leaves, tree_def = tree.flatten((args, kwargs))

    def callback(*tap_leaves: Any) -> None:
        args, kwargs = tree.unflatten(tree_def, tap_leaves)
        internal_print_generic(*args, raise_on_nan=raise_on_nan, console=console, **kwargs)

    debug.callback(callback, *leaves, ordered=True)
