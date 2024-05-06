from __future__ import annotations

from typing import cast, overload

from array_api_compat import get_namespace

from .annotations import (BooleanArray, ComplexArray, IntegralArray, JaxComplexArray, JaxRealArray,
                          RealArray)


@overload
def abs_square(x: JaxComplexArray) -> JaxRealArray:
    ...


@overload
def abs_square(x: ComplexArray) -> RealArray:
    ...


def abs_square(x: ComplexArray) -> RealArray:
    xp = get_namespace(x)
    # TODO: remove workaround when Jax is 0.4.27.
    return xp.square(x.real) + xp.square(xp.asarray(x.imag))


# TODO: Remove this when the Array API has it with broadcasting under xp.linalg.norm.
@overload
def outer_product(x: JaxRealArray, y: JaxRealArray) -> JaxRealArray:
    ...


@overload
def outer_product(x: RealArray, y: RealArray) -> RealArray:
    ...


def outer_product(x: RealArray, y: RealArray) -> RealArray:
    """Return the broadcasted outer product of a vector with itself.

    This is xp.einsum("...i,...j->...ij", x, y).
    """
    xp = get_namespace(x, y)
    xi = xp.reshape(x, (*x.shape, 1))
    yj = xp.reshape(y.conjugate(), (*y.shape[:-1], 1, y.shape[-1]))
    return xi * yj


@overload
def matrix_vector_mul(x: JaxRealArray, y: JaxRealArray) -> JaxRealArray:
    ...


@overload
def matrix_vector_mul(x: RealArray, y: RealArray) -> RealArray:
    ...


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
def matrix_dot_product(x: JaxRealArray, y: JaxRealArray) -> JaxRealArray:
    ...


@overload
def matrix_dot_product(x: RealArray, y: RealArray) -> RealArray:
    ...


def matrix_dot_product(x: RealArray, y: RealArray) -> RealArray:
    """Return the "matrix dot product" of a matrix with the outer product of a vector.

    This equals:
    * 1.19 µs: xp.einsum("...ij,...ij", x, y)
    * 1.77 µs: xp.sum(x * y, axis=(-2, -1))
    # 3.87 µs: xp.linalg.trace(xp.moveaxis(x, -2, -1) @ y)
    """  # noqa: RUF002
    xp = get_namespace(x, y)
    return xp.sum(x * y, axis=(-2, -1))


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
        return xp.true_divide(dividend, divisor)
    assert otherwise is not None
    xp = get_namespace(dividend, divisor, where, otherwise)
    dividend = xp.where(where, dividend, 1.0)
    divisor = xp.where(where, divisor, 1.0)
    quotient: ComplexArray = xp.true_divide(dividend, divisor)
    return xp.where(where, quotient, otherwise)


@overload
def divide_nonnegative(dividend: JaxRealArray, divisor: JaxRealArray) -> JaxRealArray:
    ...


@overload
def divide_nonnegative(dividend: RealArray, divisor: RealArray) -> RealArray:
    ...


def divide_nonnegative(dividend: RealArray, divisor: RealArray) -> RealArray:
    """Quotient for use with positive reals that never returns NaN.

    Returns: The quotient assuming that the dividend and divisor are nonnegative, and infinite
    whenever the divisor equals zero.
    """
    xp = get_namespace(dividend, divisor)
    return cast(RealArray, divide_where(dividend, divisor, where=divisor > 0.0, otherwise=xp.inf))


# Remove when https://github.com/scipy/scipy/pull/18605 is released.
@overload
def softplus(x: JaxRealArray) -> JaxRealArray:
    ...


@overload
def softplus(x: RealArray) -> RealArray:
    ...


def softplus(x: RealArray) -> RealArray:
    xp = get_namespace(x)
    return xp.logaddexp(xp.asarray(0.0), x)


@overload
def inverse_softplus(y: JaxRealArray) -> JaxRealArray:
    ...


@overload
def inverse_softplus(y: RealArray) -> RealArray:
    ...


def inverse_softplus(y: RealArray) -> RealArray:
    xp = get_namespace(y)
    return xp.where(y > 80.0,  # noqa: PLR2004
                    y,
                    xp.log(xp.expm1(y)))
