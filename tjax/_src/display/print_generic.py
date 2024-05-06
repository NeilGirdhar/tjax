from __future__ import annotations

from typing import Any

import numpy as np
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

    def fix(u_arg: Any, arg: Any) -> Any:
        return (np.asarray(u_arg) if isinstance(arg, np.ndarray)
                else int(u_arg.item()) if isinstance(arg, int)
                else u_arg)

    def callback(*callback_leaves: Any) -> None:
        unflattened_tree = tree.unflatten(tree_def, callback_leaves)
        v_args, v_kwargs = tree.map(fix, unflattened_tree, (args, kwargs))
        internal_print_generic(*v_args, raise_on_nan=raise_on_nan, console=console, **v_kwargs)

    debug.callback(callback, *leaves, ordered=True)
