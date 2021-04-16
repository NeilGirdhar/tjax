from functools import reduce
from numbers import Number
from operator import add
from typing import Any, Collection, Optional

import jax.numpy as jnp
import numpy as np

from .annotations import ComplexArray, ComplexNumeric, RealNumeric, ShapeLike

__all__ = ['sum_tensors', 'is_scalar', 'abs_square', 'safe_divide']


def sum_tensors(tensors: Collection[ComplexArray],
                shape: Optional[ShapeLike] = None) -> ComplexArray:
    if not tensors:
        return jnp.zeros(shape)
    return reduce(add, tensors)


def is_scalar(x: Any) -> bool:
    return isinstance(x, Number) or isinstance(x, (np.ndarray, jnp.ndarray)) and x.shape == ()


def abs_square(x: ComplexNumeric) -> RealNumeric:
    return jnp.square(x.real) + jnp.square(x.imag)


def safe_divide(numerator: ComplexNumeric, denominator: RealNumeric) -> ComplexNumeric:
    return jnp.where(denominator > 0.0, numerator / denominator, jnp.inf)
