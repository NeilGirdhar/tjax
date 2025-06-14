from __future__ import annotations

import math
import operator

import jax
import jax.numpy as jnp

from .annotations import JaxArray, JaxBooleanArray


def dynamic_tree_all(tree: object) -> JaxBooleanArray:
    """Like `jax.tree.all`, but can be used in dynamic code like jitted functions and loops."""
    return jax.tree.reduce(jnp.logical_and, tree, jnp.asarray(True))  # noqa: FBT003


def tree_sum(x: object) -> JaxArray:
    if not jax.tree.leaves(x):
        return jnp.zeros(())
    retval = jax.tree.reduce(operator.add, jax.tree.map(jnp.sum, x))
    assert isinstance(retval, JaxArray)
    assert retval.ndim == 0
    return retval


def element_count(x: object) -> int:
    def array_element_count(x: object) -> int:
        if not isinstance(x, JaxArray):
            raise TypeError
        return math.prod(x.shape)
    retval = jax.tree.reduce(operator.add, jax.tree.map(array_element_count, x), 0)
    assert isinstance(retval, int)
    return retval
