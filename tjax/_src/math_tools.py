from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
from array_api_compat import array_namespace, is_jax_namespace, is_torch_namespace
from jax.scipy.special import betaln, gammaln, loggamma, logsumexp, multigammaln

from .annotations import Array, BooleanArray, Namespace


def abs_square[T: Array](x: T) -> T:
    """Return the squared absolute value ``|x|²``, supporting complex arrays."""
    xp = array_namespace(x)
    return xp.square(x.real) + xp.square(x.imag)


# TODO: Remove when it's added to theArrayAPI:
# https://github.com/data-apis/array-api/issues/242
def outer_product[T: Array](x: T, y: T) -> T:
    """Return the broadcasted outer product of a vector with itself.

    This is xp.einsum("...i,...j->...ij", x, y).
    """
    xp = array_namespace(x, y)
    xi = xp.reshape(x, (*x.shape, 1))
    yj = xp.reshape(y.conjugate(), (*y.shape[:-1], 1, y.shape[-1]))
    return xi * yj


def matrix_vector_mul[T: Array](x: T, y: T) -> T:
    """Return the matrix-vector product.

    This is xp.einsum("...ij,...j->...i", x, y).

    Note the speed difference:
    * 14.3 µs: xp.vecdot(matrix_vector_mul(m, x), x)
    * 4.44 µs: np.einsum("...i,...ij,...j->...", x, m, x)
    """
    xp = array_namespace(x, y)
    y = xp.reshape(y, (*y.shape[:-1], 1, y.shape[-1]))
    return xp.sum(x * y, axis=-1)


def matrix_dot_product[T: Array](x: T, y: T) -> T:
    """Return the "matrix dot product" of a matrix with the outer product of a vector.

    This equals:
    * 1.19 µs: xp.einsum("...ij,...ij", x, y)
    * 1.77 µs: xp.sum(x * y, axis=(-2, -1))
    # 3.87 µs: xp.linalg.trace(xp.moveaxis(x, -2, -1) @ y)
    """
    xp = array_namespace(x, y)
    return xp.sum(x * y, axis=(-2, -1))


def divide_where[T: Array](
    dividend: T, divisor: T, *, where: BooleanArray | None = None, otherwise: T | None = None
) -> T:
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


def softplus[T: Array](x: T, /, *, xp: Namespace | None = None) -> T:
    """Softplus, which is log(1+exp(x)).

    This has asymptotic behavior exp(x) as x -> -inf and x as x -> +inf.
    """
    if xp is None:
        xp = array_namespace(x)
    return xp.logaddexp(xp.asarray(0.0), x)


def log_softplus[T: Array](x: T, /, *, xp: Namespace | None = None) -> T:
    """Log-softplus, which is log(1+log(1+exp(x))).

    This has asymptotic behavior 0 as x -> -inf and log(x) as x -> +inf.
    """
    if xp is None:
        xp = array_namespace(x)
    z = xp.asarray(0.0)
    return xp.logaddexp(z, xp.logaddexp(z, x))


def bessel_iv_ratio(v: jax.Array, x: jax.Array, /, *, iterations: int = 200) -> jax.Array:
    """Return ``I_v(x) / I_(v - 1)(x)`` for nonnegative real ``x``.

    The implementation uses a backward recurrence, which is stable for the ratio and remains fully
    differentiable with respect to both ``v`` and ``x``.
    """
    v, x = jnp.broadcast_arrays(jnp.asarray(v), jnp.asarray(x))
    q = jnp.zeros_like(v + x)

    def body(i: int, q: jax.Array) -> jax.Array:
        order = v + (iterations - i) - 1.0
        return x / (2.0 * order + x * q)

    return jax.lax.fori_loop(0, iterations, body, q)


def log_bessel_ive(v: jax.Array, x: jax.Array, /) -> jax.Array:
    """Return ``log(ive(v, x))`` for nonnegative real ``x``.

    This uses the power series for ``I_v(x)`` in log-space and then subtracts ``x`` to obtain the
    exponentially scaled quantity. The implementation is pure JAX and remains differentiable with
    respect to both ``v`` and ``x``.
    """
    v, x = jnp.broadcast_arrays(jnp.asarray(v), jnp.asarray(x))
    dtype = jnp.result_type(v, x, 1.0)
    v = v.astype(dtype)
    x = x.astype(dtype)
    safe_x = jnp.maximum(x, jnp.finfo(dtype).tiny)
    terms = jnp.arange(200, dtype=dtype).reshape((1,) * x.ndim + (200,))
    v_expanded = v[..., None]
    log_terms = (
        (2.0 * terms + v_expanded) * jnp.log(safe_x[..., None] / 2.0)
        - gammaln(terms + 1.0)
        - gammaln(terms + v_expanded + 1.0)
    )
    retval = logsumexp(log_terms, axis=-1) - x
    return jnp.where(x == 0, jnp.where(v == 0, 0.0, -jnp.inf), retval)


def complex_gammaln(a: jax.Array, /) -> jax.Array:
    """Return ``log Gamma(a)`` for real or complex arguments.

    For real ``a`` this defers to ``jax.scipy.special.gammaln``, which returns ``log|Gamma(a)|``
    (well-defined on negative non-integer reals). For complex ``a`` it routes through
    ``jax.scipy.special.loggamma``, the principal-branch complex log of Gamma. The two agree
    on positive reals.
    """
    a_arr = jnp.asarray(a)
    if jnp.iscomplexobj(a_arr):
        return loggamma(a_arr)
    return gammaln(a_arr)


def _complex_algdiv(a: jax.Array, b: jax.Array, /) -> jax.Array:  # noqa: PLR0914
    """Return ``loggamma(b) - loggamma(a + b)`` via an asymptotic series in ``1/b``.

    Complex generalization of scipy's ``algdiv``. Valid where ``|b|`` is not small and ``b``,
    ``a + b`` stay away from the negative real axis (the ``loggamma`` branch cut). The series
    has the same coefficients as the real version because it is derived from Stirling's
    expansion for ``loggamma``, which is analytic on the same domain.
    """
    c0 = 0.833333333333333e-01
    c1 = -0.277777777760991e-02
    c2 = 0.793650666825390e-03
    c3 = -0.595202931351870e-03
    c4 = 0.837308034031215e-03
    c5 = -0.165322962780713e-02
    h = a / b
    c = h / (1.0 + h)
    x = c
    d = b + (a - 0.5)
    x2 = x * x
    s3 = 1.0 + (x + x2)
    s5 = 1.0 + (x + x2 * s3)
    s7 = 1.0 + (x + x2 * s5)
    s9 = 1.0 + (x + x2 * s7)
    s11 = 1.0 + (x + x2 * s9)
    t = (1.0 / b) ** 2
    w = ((((c5 * s11 * t + c4 * s9) * t + c3 * s7) * t + c2 * s5) * t + c1 * s3) * t + c0
    w *= c / b
    u = d * jnp.log1p(a / b)
    v = a * (jnp.log(b) - 1.0)
    return jnp.where(jnp.abs(u) <= jnp.abs(v), (w - v) - u, (w - u) - v)


def complex_betaln(a: jax.Array, b: jax.Array, /) -> jax.Array:
    """Return ``log B(a, b)`` for real or complex arguments.

    For real inputs this defers to ``jax.scipy.special.betaln``. For complex inputs it uses
    ``jax.scipy.special.loggamma`` (which is defined on the complex plane), switching to an
    asymptotic series in ``1/b`` when ``|b| >= 8`` to avoid the catastrophic cancellation
    between ``loggamma(b)`` and ``loggamma(a + b)`` that the naive three-term formula suffers
    when ``|b|`` is large and ``|a|`` is small.
    """
    a_arr = jnp.asarray(a)
    b_arr = jnp.asarray(b)
    if not (jnp.iscomplexobj(a_arr) or jnp.iscomplexobj(b_arr)):
        return betaln(a_arr, b_arr)
    swap = jnp.abs(a_arr) > jnp.abs(b_arr)
    a_s = jnp.where(swap, b_arr, a_arr)
    b_s = jnp.where(swap, a_arr, b_arr)
    small_b = loggamma(a_s) + (loggamma(b_s) - loggamma(a_s + b_s))
    large_b = loggamma(a_s) + _complex_algdiv(a_s, b_s)
    return jnp.where(jnp.abs(b_s) < 8.0, small_b, large_b)  # noqa: PLR2004


def complex_multigammaln(a: jax.Array, d: int, /) -> jax.Array:
    """Return ``log Gamma_d(a) = (d(d-1)/4) log(pi) + sum_{i=0}^{d-1} loggamma(a - i/2)``.

    For real ``a`` this defers to ``jax.scipy.special.multigammaln``. For complex ``a`` it
    routes through ``jax.scipy.special.loggamma``. No cancellation issues arise because the
    definition is a sum (not a difference) of ``loggamma`` values at the shifts ``a - i/2``.
    """
    a_arr = jnp.asarray(a)
    if not jnp.iscomplexobj(a_arr):
        return multigammaln(a_arr, d)
    shifts = jnp.arange(d, dtype=a_arr.real.dtype) / 2.0
    terms = loggamma(a_arr[..., None] - shifts)
    constant = (0.25 * d * (d - 1)) * jnp.log(jnp.pi).astype(a_arr.dtype)
    return jnp.sum(terms, axis=-1) + constant


def sublinear_softplus[T: Array](x: T, maximum: T, /, *, xp: Namespace | None = None) -> T:
    """Sublinear-softplus, which is softplus(x) / (1 + softplus(x) / maximum).

    This has asymptotic behavior exp(x) as x -> -inf and maximum as x -> +inf.
    """
    if xp is None:
        xp = array_namespace(x)
    o = xp.asarray(1.0)
    sp = softplus(x, xp=xp)
    return sp / (o + sp / maximum)


def inverse_softplus[T: Array](y: T, /, *, xp: Namespace | None = None) -> T:
    """Return ``x`` such that ``softplus(x) == y``.

    Uses the stable formula ``log(expm1(y))`` for small ``y`` and the identity
    approximation ``y`` for large values (``y > 80``) where ``expm1`` overflows.
    """
    if xp is None:
        xp = array_namespace(y)
    return xp.where(
        y > 80.0,  # noqa: PLR2004
        y,
        xp.log(xp.expm1(y)),
    )


def normalize[T: Array](
    mode: Literal["l1", "l2", "max"], x: T, *, axis: tuple[int, ...] | int | None = None
) -> T:
    """Return a normalized copy of ``x`` along the given axes.

    When the norm is smaller than machine epsilon the output is set to a
    uniform distribution (l1/l2) or ones (max) rather than dividing by zero.

    Args:
        mode: Normalization strategy — ``"l1"`` divides by the L1 norm,
            ``"l2"`` divides by the L2 norm, ``"max"`` divides by the
            maximum absolute value.
        x: Input array.
        axis: Axes over which the norm is computed.  ``None`` reduces over
            all axes.
    """
    xp = array_namespace(x)
    epsilon = 10 * xp.finfo(x.dtype).eps
    match mode:
        case "l1":
            sum_x = xp.sum(xp.abs(x), axis=axis, keepdims=True)
            size = x.size / sum_x.size
            return xp.where(sum_x < epsilon, xp.ones_like(x) / size, x / sum_x)
        case "l2":
            sum_x = xp.sqrt(xp.sum(xp.square(x), axis=axis, keepdims=True))
            size = x.size / sum_x.size
            return xp.where(sum_x < epsilon, xp.ones_like(x) * xp.pow(size, -0.5), x / sum_x)
        case "max":
            sum_x = xp.max(xp.abs(x), axis=axis, keepdims=True)
            return xp.where(sum_x < epsilon, xp.ones_like(x), x / sum_x)


def stop_gradient[U](x: U, *, xp: Namespace | None = None) -> U:
    """Return ``x`` with its gradient detached, in a namespace-agnostic way.

    Dispatches to ``jax.lax.stop_gradient`` for JAX arrays, ``Tensor.detach()``
    for PyTorch tensors, and is a no-op for other namespaces.
    """
    if xp is None:
        xp = array_namespace(x)
    if is_jax_namespace(xp):
        from jax.lax import stop_gradient as sg  # noqa: PLC0415

        return sg(x)
    if is_torch_namespace(xp):
        from torch import Tensor  # noqa: PLC0415  # ty: ignore

        assert isinstance(x, Tensor)
        return x.detach()
    return x
