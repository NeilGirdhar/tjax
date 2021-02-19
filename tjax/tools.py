from functools import reduce
from numbers import Number
from operator import add
from typing import Any, Collection, Optional

import numpy as np
from chex import Array
from jax import numpy as jnp

from .annotations import ShapeLike

__all__ = ['sum_tensors', 'is_scalar', 'abs_square']


def sum_tensors(tensors: Collection[Array],
                shape: Optional[ShapeLike] = None) -> Array:
    if not tensors:
        return jnp.zeros(shape)
    return reduce(add, tensors)


def is_scalar(x: Any) -> bool:
    return isinstance(x, Number) or isinstance(x, (np.ndarray, jnp.ndarray)) and x.shape == ()


def abs_square(x: Array) -> Array:
    return jnp.square(x.real) + jnp.square(x.imag)
