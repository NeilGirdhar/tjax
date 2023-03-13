from __future__ import annotations

from numbers import Number
from typing import Any, overload

import jax.numpy as jnp
import numpy as np
from jax import float0

from .annotations import (Array, BooleanNumeric, ComplexNumeric, IntegralNumeric, JaxArray,
                          JaxComplexArray, JaxRealArray, NumpyRealArray, RealNumeric)

__all__ = ['is_scalar', 'abs_square', 'divide_where', 'divide_nonnegative', 'zero_tangent_like',
           'inverse_softplus']


def is_scalar(x: Any) -> bool:
    return isinstance(x, Number) or isinstance(x, (np.ndarray, JaxArray)) and x.shape == ()


def abs_square(x: ComplexNumeric) -> JaxRealArray:
    return jnp.square(x.real) + jnp.square(x.imag)


@overload
def divide_where(dividend: RealNumeric,
                 divisor: RealNumeric | IntegralNumeric,
                 *,
                 where: BooleanNumeric | None = None,
                 otherwise: RealNumeric | None = None) -> JaxRealArray:
    ...


@overload
def divide_where(dividend: ComplexNumeric,
                 divisor: ComplexNumeric | IntegralNumeric,
                 *,
                 where: BooleanNumeric | None = None,
                 otherwise: ComplexNumeric | None = None) -> JaxComplexArray:
    ...


def divide_where(dividend: ComplexNumeric,
                 divisor: ComplexNumeric | IntegralNumeric,
                 *,
                 where: BooleanNumeric | None = None,
                 otherwise: ComplexNumeric | None = None) -> JaxComplexArray:
    """Return the quotient or a special value when a condition is false.

    Returns: `jnp.where(where, dividend / divisor, otherwise)`, but without evaluating
    `dividend / divisor` when `where` is false.  This prevents some exceptions.
    """
    if where is None:
        assert otherwise is None
        return jnp.true_divide(dividend, divisor)
    assert otherwise is not None
    dividend = jnp.where(where, dividend, 1.0)
    divisor = jnp.where(where, divisor, 1.0)
    quotient: JaxComplexArray = jnp.true_divide(dividend, divisor)
    return jnp.where(where, quotient, otherwise)


def divide_nonnegative(dividend: RealNumeric, divisor: RealNumeric) -> JaxRealArray:
    """Quotient for use with positive reals that never returns NaN.

    Returns: The quotient assuming that the dividend and divisor are nonnegative, and infinite
    whenever the divisor equals zero.
    """
    return divide_where(dividend, divisor, where=divisor > 0.0, otherwise=jnp.inf)  # noqa: PLR2004


def zero_tangent_like(value: Array) -> NumpyRealArray:
    if jnp.issubdtype(value.dtype, jnp.inexact):
        return np.zeros_like(value)
    return np.zeros_like(value, dtype=float0)


def inverse_softplus(y: RealNumeric) -> JaxRealArray:
    return jnp.where(y > 80.0,  # noqa: PLR2004
                     y,
                     jnp.log(jnp.expm1(y)))
