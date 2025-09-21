from __future__ import annotations

import math
import operator

import jax.numpy as jnp
from jax import tree

from .annotations import JaxArray, JaxBooleanArray


def dynamic_tree_all(x: object, /) -> JaxBooleanArray:
    """Like `jax.tree.all`, but can be used in dynamic code like jitted functions and loops."""
    jax_true = jnp.asarray(True)  # noqa: FBT003
    return tree.reduce_associative(jnp.logical_and, x, identity=jax_true)


def tree_sum(x: object, /) -> JaxArray:
    retval = tree.reduce_associative(operator.add, tree.map(jnp.sum, x), identity=jnp.zeros(()))
    assert isinstance(retval, JaxArray)
    assert retval.ndim == 0
    return retval


def element_count(x: object, /) -> int:
    def array_element_count(x: object) -> int:
        if not isinstance(x, JaxArray):
            raise TypeError
        return math.prod(x.shape)
    retval = tree.reduce_associative(operator.add, tree.map(array_element_count, x), identity=0)
    assert isinstance(retval, int)
    return retval
