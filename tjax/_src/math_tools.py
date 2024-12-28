from __future__ import annotations

from typing import Literal, TypeVar, overload

import jax
import numpy as np
from array_api_compat import get_namespace

from .annotations import (BooleanArray, ComplexArray, IntegralArray, JaxBooleanArray,
                          JaxComplexArray, JaxIntegralArray, JaxRealArray, NumpyBooleanArray,
                          NumpyComplexArray, NumpyIntegralArray, NumpyRealArray, RealArray)


@overload
def abs_square(x: JaxComplexArray) -> JaxRealArray: ...
@overload
def abs_square(x: ComplexArray) -> RealArray: ...
def abs_square(x: ComplexArray) -> RealArray:
    xp = get_namespace(x)
    return xp.square(x.real) + xp.square(x.imag)


# TODO: Remove when it's added to the Array API:
# https://github.com/data-apis/array-api/issues/242
@overload
def outer_product(x: JaxRealArray, y: JaxRealArray) -> JaxRealArray: ...
@overload
def outer_product(x: RealArray, y: RealArray) -> RealArray: ...
def outer_product(x: RealArray, y: RealArray) -> RealArray:
    """Return the broadcasted outer product of a vector with itself.

    This is xp.einsum("...i,...j->...ij", x, y).
    """
    xp = get_namespace(x, y)
    xi = xp.reshape(x, (*x.shape, 1))
    yj = xp.reshape(y.conjugate(), (*y.shape[:-1], 1, y.shape[-1]))
    return xi * yj


@overload
def matrix_vector_mul(x: JaxRealArray, y: JaxRealArray) -> JaxRealArray: ...
@overload
def matrix_vector_mul(x: RealArray, y: RealArray) -> RealArray: ...
def matrix_vector_mul(x: RealArray, y: RealArray) -> RealArray:
    """Return the matrix-vector product.

    This is xp.einsum("...ij,...j->...i", x, y).

    Note the speed difference:
    * 14.3 µs: xp.vecdot(matrix_vector_mul(m, x), x)
    * 4.44 µs: np.einsum("...i,...ij,...j->...", x, m, x)
    """  # noqa: RUF002
    xp = get_namespace(x, y)
    y = xp.reshape(y, (*y.shape[:-1], 1, y.shape[-1]))
    return xp.sum(x * y, axis=-1)


@overload
def matrix_dot_product(x: JaxRealArray, y: JaxRealArray) -> JaxRealArray: ...
@overload
def matrix_dot_product(x: RealArray, y: RealArray) -> RealArray: ...
def matrix_dot_product(x: RealArray, y: RealArray) -> RealArray:
    """Return the "matrix dot product" of a matrix with the outer product of a vector.

    This equals:
    * 1.19 µs: xp.einsum("...ij,...ij", x, y)
    * 1.77 µs: xp.sum(x * y, axis=(-2, -1))
    # 3.87 µs: xp.linalg.trace(xp.moveaxis(x, -2, -1) @ y)
    """  # noqa: RUF002
    xp = get_namespace(x, y)
    return xp.sum(x * y, axis=(-2, -1))


@overload
def divide_where(dividend: JaxRealArray,
                 divisor: JaxRealArray | JaxIntegralArray,
                 *,
                 where: JaxBooleanArray | None = None,
                 otherwise: JaxRealArray | None = None) -> JaxRealArray: ...
@overload
def divide_where(dividend: NumpyRealArray,
                 divisor: NumpyRealArray | NumpyIntegralArray,
                 *,
                 where: NumpyBooleanArray | None = None,
                 otherwise: NumpyRealArray | None = None) -> NumpyRealArray: ...
@overload
def divide_where(dividend: NumpyComplexArray,
                 divisor: NumpyComplexArray | NumpyIntegralArray,
                 *,
                 where: NumpyBooleanArray | None = None,
                 otherwise: NumpyComplexArray | None = None) -> NumpyComplexArray: ...
def divide_where(dividend: ComplexArray,
                 divisor: ComplexArray | IntegralArray,
                 *,
                 where: BooleanArray | None = None,
                 otherwise: ComplexArray | None = None) -> ComplexArray:
    """Return the quotient or a special value when a condition is false.

    Returns: `xp.where(where, dividend / divisor, otherwise)`, but without evaluating
    `dividend / divisor` when `where` is false.  This prevents some exceptions.
    """
    if where is None:
        assert otherwise is None
        xp = get_namespace(dividend, divisor)
        return xp.divide(dividend, divisor)
    assert otherwise is not None
    xp = get_namespace(dividend, divisor, where, otherwise)
    dividend = xp.where(where, dividend, 1.0)
    divisor = xp.where(where, divisor, 1.0)
    quotient: ComplexArray = xp.divide(dividend, divisor)
    return xp.where(where, quotient, otherwise)


@overload
def divide_nonnegative(dividend: JaxRealArray, divisor: JaxRealArray) -> JaxRealArray: ...
@overload
def divide_nonnegative(dividend: NumpyRealArray, divisor: NumpyRealArray) -> NumpyRealArray: ...
def divide_nonnegative(dividend: RealArray, divisor: RealArray) -> RealArray:
    """Quotient for use with positive reals that never returns NaN.

    Returns: The quotient assuming that the dividend and divisor are nonnegative, and infinite
    whenever the divisor equals zero.
    """
    xp = get_namespace(dividend, divisor)
    return divide_where(dividend, divisor, where=divisor > 0.0,  # type: ignore # pyright: ignore
                        otherwise=xp.asarray(xp.inf))


# Remove when https://github.com/scipy/scipy/pull/18605 is released.
@overload
def softplus(x: JaxRealArray) -> JaxRealArray: ...
@overload
def softplus(x: RealArray) -> RealArray: ...
def softplus(x: RealArray) -> RealArray:
    xp = get_namespace(x)
    return xp.logaddexp(xp.asarray(0.0), x)


@overload
def inverse_softplus(y: JaxRealArray) -> JaxRealArray: ...
@overload
def inverse_softplus(y: RealArray) -> RealArray: ...
def inverse_softplus(y: RealArray) -> RealArray:
    xp = get_namespace(y)
    return xp.where(y > 80.0,  # noqa: PLR2004
                    y,
                    xp.log(xp.expm1(y)))


def normalize(mode: Literal['l1', 'l2', 'max'],
              x: JaxRealArray,
              *,
              axis: tuple[int, ...] | int | None = None
              ) -> JaxRealArray:
    """Returns the L1-normalized copy of x, assuming that x is nonnegative."""
    xp = get_namespace(x)
    epsilon = 10 * xp.finfo(x.dtype).eps
    match mode:
        case 'l1':
            sum_x = xp.sum(xp.abs(x), axis=axis, keepdims=True)
            size = x.size / sum_x.size
            return xp.where(sum_x < epsilon, xp.ones_like(x) / size, x / sum_x)
        case 'l2':
            sum_x = xp.sqrt(xp.sum(xp.square(x), axis=axis, keepdims=True))
            size = x.size / sum_x.size
            return xp.where(sum_x < epsilon, xp.ones_like(x) * xp.sqrt(1 / size), x / sum_x)
        case 'max':
            sum_x = xp.max(xp.abs(x), axis=axis, keepdims=True)
            return xp.where(sum_x < epsilon, xp.ones_like(x), x / sum_x)


T = TypeVar('T', bound=ComplexArray)


def create_diagonal_array(m: T) -> T:
    """A vectorized version of diagonal.

    Args:
        m: Has shape (*k, n)
    Returns: Array with shape (*k, n, n) and the elements of m on the diagonals.
    """
    xp = get_namespace(m)
    pre = m.shape[:-1]
    n = m.shape[-1]
    s = (*m.shape, n)
    retval = xp.zeros((*pre, n ** 2), dtype=m.dtype)
    for index in np.ndindex(*pre):
        target_index = (*index, slice(None, None, n + 1))
        source_values = m[*index, :]  # type: ignore[arg-type]
        if isinstance(retval, jax.Array):
            retval = retval.at[target_index].set(source_values)
        else:
            retval[target_index] = source_values
    return xp.reshape(retval, s)
