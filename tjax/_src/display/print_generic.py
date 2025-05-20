from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array, debug, tree
from rich.console import Console

from ..dataclasses.dataclass import dataclass
from .internal import internal_print_generic


@dataclass
class JaxKey:
    key_data: Array


def replace_key(leaf: Any) -> Any:
    if isinstance(leaf, Array) and jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key):
        return JaxKey(jr.key_data(leaf))
    return leaf


def print_generic(*args: object,
                  immediate: bool = False,
                  raise_on_nan: bool = True,
                  as_leaves: bool = False,
                  console: Console | None = None,
                  ) -> None:
    """Uses internal_print_generic in a tapped function.

    Args:
        immediate: Print the value immediately even if they're tracers.  Do not wait for the
            compiled function to be called.
        raise_on_nan: Assert if NaN is found in the output.
        as_leaves: Print the elements of args showing only the the tree leaves and their paths.
        console: The console that formats the output.
        args: Positional arguments to be printed.  Only dynamic arguments are allowed.
    """
    args = tree.map(replace_key, args)

    if immediate:
        internal_print_generic(*args, raise_on_nan=raise_on_nan, as_leaves=as_leaves,
                               console=console)
        return

    leaves, tree_def = tree.flatten(args)

    def fix(u_arg: Any, arg: Any) -> Any:
        return (np.asarray(u_arg) if isinstance(arg, np.ndarray)
                else int(u_arg.item()) if isinstance(arg, int)
                else u_arg)

    def callback(*callback_leaves: object) -> None:
        unflattened_tree = tree.unflatten(tree_def, callback_leaves)
        v_args = tree.map(fix, unflattened_tree, args)
        internal_print_generic(*v_args, raise_on_nan=raise_on_nan, as_leaves=as_leaves,
                               console=console)

    debug.callback(callback, *leaves, ordered=True)
