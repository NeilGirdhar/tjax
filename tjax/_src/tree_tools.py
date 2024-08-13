from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .annotations import JaxBooleanArray


def dynamic_tree_all(tree: Any) -> JaxBooleanArray:
    """Like `jax.tree.all`, but can be used in dynamic code like jitted functions and loops."""
    return jax.tree.reduce(jnp.logical_and, tree, jnp.asarray(True))
