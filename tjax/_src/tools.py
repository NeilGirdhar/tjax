from __future__ import annotations

from numbers import Number
from typing import Any, Optional, Union, overload

import jax.numpy as jnp
import numpy as np
from jax import float0

from .annotations import (Array, BooleanNumeric, ComplexArray, ComplexNumeric, IntegralNumeric,
                          JaxArray, NumpyRealArray, RealArray, RealNumeric)

__all__ = ['is_scalar', 'abs_square', 'divide_where', 'divide_nonnegative', 'zero_tangent_like',
           'inverse_softplus']


def is_scalar(x: Any) -> bool:
    return isinstance(x, Number) or isinstance(x, (np.ndarray, JaxArray)) and x.shape == ()


def abs_square(x: ComplexNumeric) -> RealArray:
    return jnp.square(x.real) + jnp.square(x.imag)


@overload
def divide_where(dividend: RealNumeric,
                 divisor: Union[RealNumeric, IntegralNumeric],
                 *,
                 where: Optional[BooleanNumeric] = None,
                 otherwise: Optional[RealNumeric] = None) -> RealArray:
    ...


@overload
def divide_where(dividend: ComplexNumeric,
                 divisor: Union[ComplexNumeric, IntegralNumeric],
                 *,
                 where: Optional[BooleanNumeric] = None,
                 otherwise: Optional[ComplexNumeric] = None) -> ComplexArray:
    ...


def divide_where(dividend: ComplexNumeric,
                 divisor: Union[ComplexNumeric, IntegralNumeric],
                 *,
                 where: Optional[BooleanNumeric] = None,
                 otherwise: Optional[ComplexNumeric] = None) -> ComplexArray:
    """
    Returns: `jnp.where(where, dividend / divisor, otherwise)`, but without evaluating
        `dividend / divisor` when `where` is false.  This prevents some exceptions.
    """
    if where is None:
        assert otherwise is None
        return jnp.true_divide(dividend, divisor)
    assert otherwise is not None
    dividend = jnp.where(where, dividend, 1.0)
    divisor = jnp.where(where, divisor, 1.0)
    quotient: ComplexArray = jnp.true_divide(dividend, divisor)
    return jnp.where(where, quotient, otherwise)


def divide_nonnegative(dividend: RealNumeric, divisor: RealNumeric) -> RealArray:
    """
    Returns: The quotient assuming that the dividend and divisor are nonnegative, and infinite
        whenever the divisor equals zero.
    """
    return divide_where(dividend, divisor, where=divisor > 0.0, otherwise=jnp.inf)


def zero_tangent_like(value: Array) -> NumpyRealArray:
    if issubclass(value.dtype.type, np.inexact):
        return np.zeros_like(value)
    return np.zeros_like(value, dtype=float0)


def inverse_softplus(y: RealNumeric) -> RealArray:
    return jnp.where(y > 80.0,
                     y,
                     jnp.log(jnp.expm1(y)))
