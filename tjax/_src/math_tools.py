from __future__ import annotations

from typing import Literal, TypeVar

from array_api_compat import array_namespace, is_jax_array, is_torch_array

from .annotations import Array, BooleanArray, Namespace

T = TypeVar('T', bound=Array)


def abs_square(x: T) -> T:
    xp = array_namespace(x)
    return xp.square(x.real) + xp.square(x.imag)  # pyright: ignore


# TODO: Remove when it's added to theArrayAPI:
# https://github.com/data-apis/array-api/issues/242
def outer_product(x: T, y: T) -> T:
    """Return the broadcasted outer product of a vector with itself.

    This is xp.einsum("...i,...j->...ij", x, y).
    """
    xp = array_namespace(x, y)
    xi = xp.reshape(x, (*x.shape, 1))
    yj = xp.reshape(y.conjugate(), (*y.shape[:-1], 1, y.shape[-1]))
    return xi * yj


def matrix_vector_mul(x: T, y: T) -> T:
    """Return the matrix-vector product.

    This is xp.einsum("...ij,...j->...i", x, y).

    Note the speed difference:
    * 14.3 µs: xp.vecdot(matrix_vector_mul(m, x), x)
    * 4.44 µs: np.einsum("...i,...ij,...j->...", x, m, x)
    """  # noqa: RUF002
    xp = array_namespace(x, y)
    y = xp.reshape(y, (*y.shape[:-1], 1, y.shape[-1]))
    return xp.sum(x * y, axis=-1)


def matrix_dot_product(x: T, y: T) -> T:
    """Return the "matrix dot product" of a matrix with the outer product of a vector.

    This equals:
    * 1.19 µs: xp.einsum("...ij,...ij", x, y)
    * 1.77 µs: xp.sum(x * y, axis=(-2, -1))
    # 3.87 µs: xp.linalg.trace(xp.moveaxis(x, -2, -1) @ y)
    """  # noqa: RUF002
    xp = array_namespace(x, y)
    return xp.sum(x * y, axis=(-2, -1))


def divide_where(dividend: T,
                 divisor: T,
                 *,
                 where: BooleanArray | None = None,
                 otherwise: T | None = None) -> T:
    """Return the quotient or a special value when a condition is false.

    Returns: `xp.where(where, dividend / divisor, otherwise)`, but without evaluating
    `dividend / divisor` when `where` is false.  This prevents both infinite cotangents, and
    some exceptions.
    """
    if where is None:
        assert otherwise is None
        xp = array_namespace(dividend, divisor)
        return xp.divide(dividend, divisor)
    assert otherwise is not None
    xp = array_namespace(dividend, divisor, where, otherwise)
    dividend = xp.where(where, dividend, 1.0)
    divisor = xp.where(where, divisor, 1.0)
    quotient: T = xp.divide(dividend, divisor)
    return xp.where(where, quotient, otherwise)


# Remove when https://github.com/scipy/scipy/pull/18605 is released.
def softplus(x: T, /, *, xp: Namespace | None = None) -> T:
    """Softplus, which is log(1+exp(x)).

    This has asymptotic behavior exp(x) as x -> -inf and x as x -> +inf.
    """
    if xp is None:
        xp = array_namespace(x)
    return xp.logaddexp(xp.asarray(0.0), x)


def log_softplus(x: T, /, *, xp: Namespace | None = None) -> T:
    """Log-softplus, which is log(1+log(1+exp(x))).

    This has asymptotic behavior 0 as x -> -inf and log(x) as x -> +inf.
    """
    if xp is None:
        xp = array_namespace(x)
    z = xp.asarray(0.0)
    return xp.logaddexp(z, xp.logaddexp(z, x))


def sublinear_softplus(x: T, maximum: T, /, *, xp: Namespace | None = None) -> T:
    """Sublinear-softplus, which is softplus(x) / (1 + softplus(x) / maximum).

    This has asymptotic behavior exp(x) as x -> -inf and maximum as x -> +inf.
    """
    if xp is None:
        xp = array_namespace(x)
    o = xp.asarray(1.0)
    sp = softplus(x, xp=xp)
    return sp / (o + sp / maximum)


def inverse_softplus(y: T, /, *, xp: Namespace | None = None) -> T:
    if xp is None:
        xp = array_namespace(y)
    return xp.where(y > 80.0,  # noqa: PLR2004
                    y,
                    xp.log(xp.expm1(y)))


def normalize(mode: Literal['l1', 'l2', 'max'],
              x: T,
              *,
              axis: tuple[int, ...] | int | None = None
              ) -> T:
    """Returns the L1-normalized copy of x, assuming that x is nonnegative."""
    xp = array_namespace(x)
    epsilon = 10 * xp.finfo(x.dtype).eps
    match mode:
        case 'l1':
            sum_x = xp.sum(xp.abs(x), axis=axis, keepdims=True)
            size = x.size / sum_x.size
            return xp.where(sum_x < epsilon, xp.ones_like(x) / size, x / sum_x)
        case 'l2':
            sum_x = xp.sqrt(xp.sum(xp.square(x), axis=axis, keepdims=True))
            size = x.size / sum_x.size
            return xp.where(sum_x < epsilon, xp.ones_like(x) * xp.pow(size, -0.5), x / sum_x)
        case 'max':
            sum_x = xp.max(xp.abs(x), axis=axis, keepdims=True)
            return xp.where(sum_x < epsilon, xp.ones_like(x), x / sum_x)


U = TypeVar('U')


def stop_gradient(x: U, *, xp: Namespace | None = None) -> U:
    if xp is None:
        xp = array_namespace(x)
    if is_jax_array(xp):
        from jax.lax import stop_gradient as sg  # noqa: PLC0415
        return sg(x)
    if is_torch_array(xp):
        from torch import Tensor  # noqa: PLC0415
        assert isinstance(x, Tensor)
        return x.detach()  # pyright: ignore
    return x
