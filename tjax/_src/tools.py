from functools import reduce
from numbers import Number
from operator import add
from typing import Any, Collection, Optional, Union, overload

import jax.numpy as jnp
import numpy as np

from .annotations import (BooleanNumeric, ComplexArray, ComplexNumeric, IntegralNumeric,
                          RealArray, RealNumeric, ShapeLike)

__all__ = ['sum_tensors', 'is_scalar', 'abs_square', 'divide_nonnegative']


@overload
def sum_tensors(tensors: Collection[IntegralNumeric],
                shape: Optional[ShapeLike] = None) -> IntegralNumeric:
    ...


@overload
def sum_tensors(tensors: Collection[RealNumeric],
                shape: Optional[ShapeLike] = None) -> RealNumeric:
    ...


@overload
def sum_tensors(tensors: Collection[ComplexNumeric],
                shape: Optional[ShapeLike] = None) -> ComplexNumeric:
    ...


def sum_tensors(tensors: Collection[Union[IntegralNumeric, ComplexNumeric]],
                shape: Optional[ShapeLike] = None) -> Union[IntegralNumeric, ComplexNumeric]:
    if not tensors:
        return jnp.zeros(shape)
    return reduce(add, tensors)


def is_scalar(x: Any) -> bool:
    return isinstance(x, Number) or isinstance(x, (np.ndarray, jnp.ndarray)) and x.shape == ()


def abs_square(x: ComplexNumeric) -> RealNumeric:
    return jnp.square(x.real) + jnp.square(x.imag)


@overload
def divide_where(dividend: RealNumeric,
                 divisor: Union[RealNumeric, IntegralNumeric],
                 *,
                 where: Optional[BooleanNumeric] = None,
                 otherwise: Optional[RealNumeric] = None) -> RealNumeric:
    ...


@overload
def divide_where(dividend: ComplexNumeric,
                 divisor: Union[ComplexNumeric, IntegralNumeric],
                 *,
                 where: Optional[BooleanNumeric] = None,
                 otherwise: Optional[ComplexNumeric] = None) -> ComplexNumeric:
    ...


def divide_where(dividend: ComplexNumeric,
                 divisor: Union[ComplexNumeric, IntegralNumeric],
                 *,
                 where: Optional[BooleanNumeric] = None,
                 otherwise: Optional[ComplexNumeric] = None) -> ComplexNumeric:
    """
    Returns: `jnp.where(where, dividend / divisor, otherwise)`, but without evaluating
        `dividend / divisor` when `where` is false.  This prevents some exceptions.
    """
    if where is None:
        return jnp.true_divide(dividend, divisor)
    dividend = jnp.where(where, dividend, 1.0)
    divisor = jnp.where(where, divisor, 1.0)
    quotient = jnp.true_divide(dividend, divisor)
    return jnp.where(where, quotient, otherwise)


def divide_nonnegative(dividend: RealNumeric, divisor: RealNumeric) -> RealNumeric:
    """
    Returns: The quotient assuming that the dividend and divisor are nonnegative, and infinite
        whenever the divisor equals zero.
    """
    return divide_where(dividend, divisor, where=divisor > 0.0, otherwise=jnp.inf)
