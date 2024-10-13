# ruff: noqa: ANN001, ANN002, ANN202, ANN201, TRY003, EM101, FBT001, PLR0914, PLR2004, FBT002
# Copyright 2020 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Implements bessel functions."""

from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import custom_jvp, lax
from jax.typing import DTypeLike

from ._src.annotations import BooleanArray, RealArray
from .dataclasses import dataclass

__all__ = [
    'bessel_iv_ratio',
    'bessel_ive',
    'bessel_kve',
    'log_bessel_ive',
    'log_bessel_kve',
]


def as_numpy_dtype(dtype_like: DTypeLike) -> type:
    dtype = jnp.dtype(dtype_like)
    if hasattr(dtype, 'as_numpy_dtype'):
        return dtype.as_numpy_dtype  # pyright: ignore
    return dtype.type


def log1mexp(x: RealArray) -> RealArray:
    """Compute `log(1 - exp(-|x|))` elementwise in a numerically stable way.

    Args:
        x: Float `Tensor`.
        name: Python `str` name prefixed to Ops created by this function.
            Default value: `None` (i.e., `'log1mexp'`).

    Returns:
        log1mexp: Float `Tensor` of `log1mexp(x)`.

    #### References

    [1]: Machler, Martin. Accurately computing log(1 - exp(-|a|))
             https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    """
    x = jnp.abs(x)
    return jnp.where(
            # This switching point is recommended in [1].
            x < np.log(2), jnp.log(-jnp.expm1(-x)),
            jnp.log1p(-jnp.exp(-x)))


def log_sub_exp(x: RealArray, y: RealArray, *, return_sign: bool = False):
    """Compute `log(exp(max(x, y)) - exp(min(x, y)))` in a numerically stable way.

    Use `return_sign=True` unless `x >= y`, since we can't represent a negative in
    log-space.

    Args:
        x: Float `Tensor` broadcastable with `y`.
        y: Float `Tensor` broadcastable with `x`.
        return_sign: Whether or not to return the second output value `sign`. If
            it is known that `x >= y`, this is unnecessary.
        name: Python `str` name prefixed to Ops created by this function.
            Default value: `None` (i.e., `'log_sub_exp'`).

    Returns:
        logsubexp: Float `Tensor` of `log(exp(max(x, y)) - exp(min(x, y)))`.
        sign: Float `Tensor` +/-1 indicating the sign of `exp(x) - exp(y)`.
    """
    dtype = jnp.result_type(x, y)
    x = jnp.asarray(x, dtype=dtype)
    y = jnp.asarray(y, dtype=dtype)
    larger = jnp.maximum(x, y)
    smaller = jnp.minimum(x, y)
    result = larger + log1mexp(jnp.maximum(larger - smaller, 0))
    if return_sign:
        ones = jnp.ones((), result.dtype)
        return result, jnp.where(x < y, -ones, ones)
    return result


# def fix_gradient_for_broadcasting(primals, grads):
#     """Ensure `grads` have same shape as `primals`."""
#     if len(primals) != len(grads):
#         raise ValueError('Expected same number of `x` and `grads`')
#     if all(x.shape == primals[0].shape for x in primals):
#         return grads
#     # Compute the leave one out broadcast shapes, and use that to compute
#     # the axes.
#     new_grads = []
#     primal_shapes = [jnp.shape(x) for x in primals]
#     for i in range(len(primals)):
#         loo_primal_shapes = primal_shapes[:i] + primal_shapes[i + 1:]
#         x_shape = jnp.shape(primals[i])
#         rx, _ = tf.raw_ops.BroadcastGradientArgs(
#                 s0=x_shape, s1=loo_primal_shapes)
#         new_grads.append(jnp.reshape(jnp.sum(grads[i], axis=rx), shape=x_shape))
#     return new_grads


def _sqrt1px2(x: RealArray) -> RealArray:
    return jnp.where(
            jnp.abs(x) * np.sqrt(np.finfo(x.dtype).eps) <= 1.,
            jnp.exp(0.5 * jnp.log1p(jnp.square(x))),
            jnp.abs(x))


def _compute_general_continued_fraction(
        max_iterations: int,
        numerator_denominator_args_list: Any,
        tolerance: RealArray | float | None = None,
        partial_numerator_fn: Any = None,
        partial_denominator_fn: Any = None,
        dtype: DTypeLike = jnp.float32,
        ) -> RealArray:
    """Compute a general continued fraction.

    Given at least one of `partial_numerator_fn` and `partial_denominator_fn`, compute the continued
    fraction associated with it via the forward recurrence.

    Let `a_i = partial_numerator_fn` and `b_i = partial_denominator_fn`. Then, this evaluates the
    infinite continued fraction:

    ```result = a_1 / (b_1 + a_2 / (b_2 + a_3 / (b_3 .....)))```.

    If `partial_numerator_fn` or `partial_denominator_fn` are not given, then
    `a_i` (respectively `b_i`) are assumed to be 1. However one must be given.

    NOTE: Use this with caution. Forward recursion doesn't have numerical stability guarantees,
    compared to backward recursion.


    Args:
        max_iterations: Integer `Tensor` specifying the maximum number of terms to
            use.
        numerator_denominator_args_list: Arguments to pass in to
            `partial_numerator_fn` and `partial_denominator_fn`.
        tolerance: Float `Tensor` specifying the maximum acceptable tolerance
            between convergents. If unset, convergence is dictated by the number
            of iterations.
            Default value: `None`.
        partial_numerator_fn: Python callable that takes in as its first argument
            the current iteration count (an integer >= 1), and a list of *args, and
            returns a `Tensor`. These are used as partial numerators for the
            continued fraction.
            Default value: `None`.
        partial_denominator_fn: Python callable that takes in as its first argument
            the current iteration count (an integer >= 1), and a list of *args, and
            returns a `Tensor`. These are used as partial denominators for the
            continued fraction.
            Default value: `None`.
        dtype: The default dtype of the continued fraction. Default: `float32`.

    Returns:
        Continued fraction computed to `max_iterations` iterations and/or
        up to absolute error `tolerance`.

    #### References
    [1]: Walter Gautschi and Josef Slavik. On the Computation of Modified Bessel Function Ratios.
         http://www.jstor.com/stable/2006491
    """
    dtype = jnp.result_type(*numerator_denominator_args_list, dtype)

    if (partial_numerator_fn is None) and (partial_denominator_fn is None):
        raise ValueError('Expect one of `partial_numerator_fn` and '
                         '`partial_denominator_fn` to be set.')

    @dataclass
    class Carry:
        unused_should_stop: BooleanArray
        numerator: RealArray
        previous_numerator: RealArray
        denominator: RealArray
        previous_denominator: RealArray
        iteration_count: RealArray

    def _continued_fraction_one_step(c: Carry) -> Carry:
        partial_denominator = 1.
        if partial_denominator_fn:
            partial_denominator = partial_denominator_fn(
                    c.iteration_count, *numerator_denominator_args_list)
        new_numerator = partial_denominator * c.numerator
        new_denominator = partial_denominator * c.denominator

        partial_numerator = 1.
        if partial_numerator_fn:
            partial_numerator = partial_numerator_fn(
                    c.iteration_count, *numerator_denominator_args_list)
        new_numerator += partial_numerator * c.previous_numerator
        new_denominator += partial_numerator * c.previous_denominator

        should_stop_next = c.iteration_count > max_iterations

        if tolerance is not None:
            # We can use a more efficient computation when the partial numerators
            # are 1.
            if partial_numerator_fn is None:
                # We now want to compute to relative error between the fraction at
                # this iteration, vs. the previous iteration.
                # Let h_i be the numerator and k_i the denominator, and a_i be the
                # i-th term.
                # h_i / k_i - h_{i-1} / k_{i-1} =
                # (h_i * k_{i - 1} - h_{i - 1} * k_i) / (k_i * k_{i - 1}) =
                # ((a_i h_{i - 1} + h_{i - 2}) * k_{i - 1} -
                # (a_i k_{i - 1} + k_{i - 2}) * h_{i - 1}) / (k_i * k_{i - 1}) =
                # -(h_{i - 1} * k_{i - 2} - h_{i - 2} * k_{i - 1}) / (k_i * k_{i - 1})
                # This suggests we should prove something about the numerator
                # inductively, and indeed
                # (h_i * k_{i - 1} - h_{i - 1} * k_i) = (-1)**i
                delta = jnp.reciprocal(new_denominator * c.denominator)
            # We actually need to compute the difference of fractions.
            else:
                delta = new_numerator / new_denominator - c.numerator / c.denominator

            converged = jnp.abs(delta) <= tolerance
            should_stop_next |= jnp.all(converged)
        return Carry(should_stop_next,
                     new_numerator,
                     c.numerator,
                     new_denominator,
                     c.denominator,
                     c.iteration_count + 1.)

    # This is to infer the correct shape of tensors
    if partial_denominator_fn:
        term = partial_denominator_fn(1., *numerator_denominator_args_list)
    else:
        term = partial_numerator_fn(1., *numerator_denominator_args_list)

    zeroth_numerator = jnp.ones_like(term, dtype=dtype)
    zeroth_denominator = jnp.zeros_like(term, dtype=dtype)
    first_numerator = jnp.zeros_like(term, dtype=dtype)
    first_denominator = jnp.ones_like(term, dtype=dtype)

    results = lax.while_loop(
            cond_fun=lambda carry: ~carry.unused_should_stop,
            body_fun=_continued_fraction_one_step,
            init_val=Carry(jnp.asarray(False),
                           first_numerator,
                           zeroth_numerator,
                           first_denominator,
                           zeroth_denominator,
                           jnp.asarray(1., dtype=dtype)))
    return results.numerator / results.denominator


def _bessel_iv_ratio_naive(v: RealArray, z: RealArray) -> RealArray:
    """Compute bessel_iv_ratio(v, z)."""
    dtype = jnp.result_type(v, z, jnp.float32)
    v = jnp.asarray(v, dtype=dtype)
    z = jnp.asarray(z, dtype=dtype)

    # I(v, z) == I(-v, z) when v is an integer.
    v_is_integer = jnp.equal(jnp.floor(v), v)
    v = jnp.where((v < 0.) & v_is_integer, -v, v)

    np_finfo = np.finfo(dtype)
    tolerance = jnp.asarray(np_finfo.resolution, dtype=dtype)

    safe_to_use_perron = z > v

    @dataclass
    class Carry:
        iteration_count: RealArray
        cf: RealArray

    def gauss_term_fn(iteration_count: RealArray, v: RealArray, z: RealArray) -> RealArray:
        """Terms for the Gauss continued fraction."""
        return jnp.square(z) / 4. / (
                (v + iteration_count - 1) * (v + iteration_count))

    # The Gauss continued fraction converges faster for z < v.
    # For z > v, set z to something much less than v.
    safe_z_less_v = jnp.where(safe_to_use_perron, v / 1000., z)

    # We use forward recurrence for the Gauss continued fraction.
    # This is so that we can do early termination.
    # There are a few reasons why this doesn't overflow:
    # * All partial numerators / denominators are positive.
    # * Partial numerators approach zero as 1 / n**2, where
    #     n is the iteration count.
    # * All partial numerators are less than 1.
    # Combined with the recurrence, this ensures no overflow.
    # as the number of iterations -> infinity.
    gauss_cf = _compute_general_continued_fraction(
            # Use a max of 200 steps. Almost always we will be much less
            # than this.
            200, [v, safe_z_less_v], tolerance=tolerance,
            partial_numerator_fn=gauss_term_fn)
    # Add the zeroth term for the Gauss continued fraction.
    gauss_cf = jnp.reciprocal((1. + gauss_cf) * 2. * v / z)

    # For the Perron CF we use the backward recurrence. This is because
    # generally the backward recurrence is more numerically stable
    # than forward recurrence, especially with negative terms.
    # We use a flat 50 steps. Anecdotally, for z > v, convergence is
    # much faster than that.

    # The Perron continued fraction converges much faster for z >> v.
    # For z < v, set z to something much greater than v.
    safe_z_greater_v = jnp.where(~safe_to_use_perron, 1000. * v, z)

    def perron_term_fn(iteration_count: RealArray, v: RealArray, z: RealArray) -> RealArray:
        """Terms for the Perron continued fraction."""
        return -0.5 * z * (v + iteration_count - 0.5) / (
                (v + z + (iteration_count - 1.) / 2.) *
                (v + z + iteration_count / 2.))

    total_perron_iteration_count = 50

    def _backward_cf_one_step(c: Carry) -> Carry:
        cf = perron_term_fn(
                total_perron_iteration_count - c.iteration_count,
                v, safe_z_greater_v) / (1. + c.cf)
        return Carry(c.iteration_count + 1., cf)

    # For the Perron CF, we omit the first numerator because it
    # has a different form.

    perron_carry = lax.while_loop(
            cond_fun=lambda c: c.iteration_count < total_perron_iteration_count - 1,
            body_fun=_backward_cf_one_step,
            # Use 50 iterations. Empirically, the Perron continued fraction converges much faster
            # than this.
            init_val=Carry(jnp.asarray(0., dtype=dtype),
                           jnp.zeros_like(safe_z_greater_v)))
    first_term = -0.5 * z * (v + 0.5) / ((v + z / 2.) * (v + z + 0.5))

    perron_cf = first_term / (1. + perron_carry.cf)

    # Add the zeroth term for the Perron continued fraction.
    perron_zeroth_term = (z + 2 * v) / z
    perron_cf = jnp.reciprocal(perron_zeroth_term * (1. + perron_cf))
    return jnp.where(safe_to_use_perron, perron_cf, gauss_cf)


# def _bessel_iv_ratio_fwd(v: RealArray, z: RealArray) -> RealArray:
#     """Compute output, aux (collaborates with _bessel_iv_ratio_bwd)."""
#     output = _bessel_iv_ratio_naive(v, z)
#     return output, (v, z)


def _bessel_iv_ratio_partial(v: RealArray, z: RealArray):
    """Computes the derivative of the ratio elementwise with respect to z.

    For shorthand, let `I(v) = I(v, z)`, `R(v) = I(v, z) / I(v - 1, z)`
    ```
    R'(v) = (I'(v)I(v - 1) - I(v)I'(v - 1)) / I(v - 1) ** 2
                 = 0.5 * ((I(v - 1) + I(v + 1))I(v - 1) - I(v)(
                            I(v) + I(v - 2))) / I(v - 1) ** 2
                 = 0.5 * (1. + I(v + 1) / I(v - 1) - (I(v) / I(v - 1)) ** 2 - (
                            I(v) / I(v - 1)) * (I(v - 2) / I(v - 1)))
                 = 0.5 * (1. + R(v + 1) * R(v) - R(v) ** 2 - R(v) / R(v - 1))
                 = 0.5 * (1. + R(v) * (R(v + 1) - R(v) - 1. / R(v - 1)))
    ```
    To avoid computing R(v - 1) when v <= 1 (which is not valid),
    we can rewrite `I(v - 2) = 2 (v - 1) / z * I(v - 1) + I(v)`.
    Thus the last term becomes:
    ```
    -1. / R(v - 1) = -I(v - 2) / I(v - 1) = -2 (v - 1) / z - R(v)
    ```

    Args:
        v: A Tensor with type `float32` or `float64`.
        z: A Tensor with type `float32` or `float64`.

    Returns:
        A Tensor with same shape and dtype as `z`.
    """
    result = _bessel_iv_ratio_custom_gradient(v, z)
    return 0.5 * (1. + result * (_bessel_iv_ratio_custom_gradient(v + 1., z) -
                                 2. * result - 2. * (v - 1) / z))


# def _bessel_iv_ratio_bwd(aux: RealArray, g: RealArray) -> tuple[None, RealArray]:
#     """Reverse mode impl for bessel_iv_ratio."""
#     v, z = aux
#     pz = _bessel_iv_ratio_partial(v, z)
#     grad_z = pz * g
#     # _, grad_z = tfp_math.fix_gradient_for_broadcasting(
#     #         [v, z], [jnp.ones_like(grad_z), grad_z])
#     return None, grad_z


def _bessel_iv_ratio_jvp(primals: Any, tangents: Any) -> tuple[RealArray, RealArray]:
    """Computes JVP for bessel_iv_ratio (supports JAX custom derivative)."""
    v, z = primals
    _, dz = tangents

    # TODO(https://github.com/google/jax/issues/3768): eliminate broadcast_to?
    bc_shp = jnp.broadcast_shapes(jnp.shape(v), jnp.shape(dz))
    dz = jnp.broadcast_to(dz, bc_shp)

    x = _bessel_iv_ratio_naive(v, z)
    pz = _bessel_iv_ratio_partial(v, z)

    # `bessel_iv_ratio` does not have gradients with respect to `v`, and thus
    # this `JVP` rule matches TF.
    # Ideally, it would be nice to throw an exception when taking gradients of
    # in JAX mode, but this is not possible at the moment with `custom_jvp`.
    # See https://github.com/google/jax/issues/5913 for details.
    # TODO(https://github.com/google/jax/issues/5913): Define vjp for v.

    return x, pz * dz


@custom_jvp
def _bessel_iv_ratio_custom_gradient(v: RealArray, z: RealArray) -> RealArray:
    return _bessel_iv_ratio_naive(v, z)


_bessel_iv_ratio_custom_gradient.defjvp(_bessel_iv_ratio_jvp)


def bessel_iv_ratio(v: RealArray, z: RealArray):
    """Computes `I_{v} (z) / I_{v - 1} (z)` in a numerically stable way.

    Let I(v, z) be the modified bessel function of the first kind. This computes
    the ratio of I(v, z) / I(v - 1, z). This can be more numerically stable
    and faster than computing the ratio directly.

    This uses a continued fraction approximation attributed to Gauss for
    computing this quantity in the limit where z <= v, and a continued fraction
    approximation attributed to Perron for z > v.

    Args:
        v: value for which `I_{v}(z) / I_{v - 1}(z)` should be computed. Expect
            v > 0.
        z: value for which `I_{v}(z) / I_{v - 1}(z)` should be computed. Expect
            z > 0.
        name: A name for the operation (optional).
            Default value: `None` (i.e., 'bessel_iv_ratio').

    Returns:
        I(v, z) / I(v - 1, z).

    #### References
    [1]: Walter Gautschi and Josef Slavik. On the Computation of Modified
             Bessel Function Ratios. http://www.jstor.com/stable/2006491
    """
    dtype = jnp.result_type(v, z, jnp.float32)
    v = jnp.asarray(v, dtype=dtype)
    z = jnp.asarray(z, dtype=dtype)
    return _bessel_iv_ratio_custom_gradient(v, z)


# Used for the polynomial coefficients parameterizing Olver's expansion.
_ASYMPTOTIC_OLVER_EXPANSION_COEFFICIENTS = [
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         -0.20833333333333334, 0., 0.125, 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0.3342013888888889, 0.,
         -0.40104166666666669, 0., 0.0703125, 0., 0.0],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., -1.0258125964506173, 0., 1.8464626736111112,
         0., -0.89121093750000002, 0., 0.0732421875, 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 4.6695844234262474, 0., -11.207002616222995, 0.,
         8.78912353515625, 0., -2.3640869140624998, 0., 0.112152099609375,
         0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         -28.212072558200244, 0., 84.636217674600744, 0., -91.818241543240035,
         0., 42.534998745388457, 0., -7.3687943594796312, 0., 0.22710800170898438,
         0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 212.5701300392171, 0.,
         -765.25246814118157, 0., 1059.9904525279999, 0., -699.57962737613275,
         0., 218.19051174421159, 0., -26.491430486951554, 0., 0.57250142097473145,
         0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., -1919.4576623184068, 0.,
         8061.7221817373083, 0., -13586.550006434136, 0., 11655.393336864536,
         0., -5305.6469786134048, 0., 1200.9029132163525, 0.,
         -108.09091978839464, 0., 1.7277275025844574, 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 20204.291330966149, 0., -96980.598388637503, 0.,
         192547.0012325315, 0., -203400.17728041555, 0., 122200.46498301747,
         0., -41192.654968897557, 0., 7109.5143024893641, 0.,
         -493.915304773088, 0., 6.074042001273483, 0., 0., 0., 0., 0.,
         0., 0., 0.],
        [0., 0., 0., -242919.18790055133, 0., 1311763.6146629769, 0.,
         -2998015.9185381061, 0., 3763271.2976564039, 0., -2813563.2265865342, 0.,
         1268365.2733216248, 0., -331645.17248456361, 0., 45218.768981362737, 0.,
         -2499.8304818112092, 0., 24.380529699556064, 0., 0., 0., 0., 0.,
         0., 0., 0., 0.0],
        [3284469.8530720375, 0., -19706819.11843222, 0., 50952602.492664628,
         0., -74105148.211532637, 0., 66344512.274729028, 0., -37567176.660763353,
         0., 13288767.166421819, 0., -2785618.1280864552, 0., 308186.40461266245,
         0., -13886.089753717039, 0., 110.01714026924674, 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0.]
]


def _olver_asymptotic_uniform(v: RealArray, z: RealArray,
                              *,
                              output_log_space: bool = False
                              ) -> tuple[RealArray, RealArray]:
    """Use Olver's uniform asymptotic expansion for the Bessel function.

    Olver's uniform asymptotic expansion [1] is specified by

    `I_v(v, v * z) ~ f(a, v) * sum_k U_k(1 / sqrt(1 + z^2)) / v^k`
    `K_v(v, v * z) ~ f(a, v) * sum_k (-1) ** k * U_k(1 / sqrt(1 + z^2)) / v^k`
    where

    * `f(a, v) = `exp(v * a) / (sqrt(2 * pi * v) * (1 + z^2)^0.25)`
    * `U_k(z)` are polynomials that are given in [2]. We use the first
    10 polynomials.

    #### References
    [1]: Digital Library of Mathematical Functions: https://dlmf.nist.gov/10.41
    [2]: F. Olver, Tables for Bessel Functions of Moderate or Large Orders.
             National Physical Laboratory Mathematical Tables, Vol. 6.
             Department of Scientific and Industrial Research

    Args:
        v: value for which `I_{v}(z)` and `K_{v}(z) should be computed.
        z: value for which `I_{v}(z)` and `K_{v}(z) should be computed.
        output_log_space: `bool`. If `True`, output is in log-space.
            Default value: `False`.
        name: A name for the operation (optional).
            Default value: `None` (i.e., 'olver_asymptotic_uniform').

    Returns:
        ive, kve: Asymptotic approximations to the modified bessel functions of the
            first and second kind.
    """
    v_abs = jnp.abs(v)
    w = z / v_abs
    t = jnp.reciprocal(_sqrt1px2(w))
    n_ufactors = len(_ASYMPTOTIC_OLVER_EXPANSION_COEFFICIENTS)

    divisor = v_abs
    ive_sum = 1.
    kve_sum = 1.

    # Note the polynomials have properties of oddness and evenness so that
    # could be taken advantage of when doing evaluation. For simplicity,
    # we naively sum using Horner's method.
    for i in range(n_ufactors):
        coeff = 0.
        for c in _ASYMPTOTIC_OLVER_EXPANSION_COEFFICIENTS[i]:
            coeff = coeff * t + c
        term = coeff / divisor
        ive_sum += term
        kve_sum += term if i % 2 == 1 else -term
        divisor *= v_abs

    # This is modified from the original impl to be more numerically stable
    # since we are subtracting off x.
    shared_prefactor = (jnp.reciprocal(_sqrt1px2(w) + w) + jnp.log(w)
                                            - jnp.log1p(jnp.reciprocal(t)))
    log_i_prefactor = 0.5 * jnp.log(
            t / (2 * np.pi * v_abs)) + v_abs * shared_prefactor

    # Not the same here since they will have the same sign.
    log_k_prefactor = 0.5 * jnp.log(
            np.pi * t / (2 * v_abs)) - v_abs * shared_prefactor

    log_kve = log_k_prefactor + jnp.log(kve_sum)
    log_ive = log_i_prefactor + jnp.log(ive_sum)

    # We need to add a correction term for negative v.
    negative_v_correction = log_kve - 2. * z
    n = jnp.round(v)
    u = v - n
    coeff = 2 / np.pi * jnp.sin(np.pi * u)
    coeff = (1. - 2. * jnp.mod(n, 2.)) * coeff

    lse, sign = log_sub_exp(
            log_ive,
            negative_v_correction + jnp.log(jnp.abs(coeff)),
            return_sign=True)
    sign = jnp.where(coeff < 0., sign, 1.)

    log_ive_negative_v = jnp.where(
            coeff < 0.,
            lse,
            jnp.logaddexp(
                    log_ive, negative_v_correction + jnp.log(jnp.abs(coeff))))

    if output_log_space:
        log_ive = jnp.where(v >= 0., log_ive, log_ive_negative_v)
        return log_ive, log_kve

    ive = jnp.where(
            v >= 0.,
            jnp.exp(log_ive),
            sign * jnp.exp(log_ive_negative_v))
    return ive, jnp.exp(log_kve)


def _evaluate_temme_coeffs(v: RealArray):
    """Numerically stable computation of difference of gammas."""
    # This function computes the following quantities:
    # coeff1 = (1 / Gamma(1 - v) - 1 / Gamma(1 + v)) / 2v
    # coeff2 = (1 / Gamma(1 - v) + 1 / Gamma(1 + v)) / 2
    # gamma1mv = 1 / Gamma(1 - v)
    # gamma1pv = 1 / Gamma(1 + v)
    # Naive computation of the above two coefficients leads to
    # catastrophic cancellations. The below function computes
    # Chebyshev expansions to `coeff1` and `coeff2`.

    # Stable evaluation of the coefficients for the Temme power series.
    # We refer to [1] for the numerical evaluation
    # [1] Numerical Recipes in C. The Art of Scientific Computing,
    #     2nd Edition, 1992

    # These are Chebyshev expansion coefficients defined in 6.7.18 in [1].
    coeff1_coeffs = [-1.142022680371168e0, 6.5165112670737e-3,
                                     3.087090173086e-4, -3.4706269649e-6, 6.9437664e-9,
                                     3.67795e-11, -1.356e-13]
    coeff2_coeffs = [1.843740587300905e0, -7.68528408447867e-2,
                                     1.2719271366546e-3, -4.9717367042e-6, -3.31261198e-8,
                                     2.423096e-10, -1.702e-13, -1.49e-15]
    w = 8 * jnp.square(v) - 1.

    # Use Clenshaw's recurrence for evaluating the Chebyshev polynomials
    # associated to the coefficients.
    y = 2 * w

    prev = 0.
    current = 0.
    for i in reversed(range(1, len(coeff1_coeffs))):
        temp = current
        current = y * current - prev + coeff1_coeffs[i]
        prev = temp
    coeff1 = w * current - prev + 0.5 * coeff1_coeffs[0]

    prev = 0.
    current = 0.
    for i in reversed(range(1, len(coeff2_coeffs))):
        temp = current
        current = y * current - prev + coeff2_coeffs[i]
        prev = temp
    coeff2 = w * current - prev + 0.5 * coeff2_coeffs[0]
    gamma1pv = coeff2 - v * coeff1
    gamma1mv = coeff2 + v * coeff1
    return coeff1, coeff2, gamma1pv, gamma1mv


def _temme_series(v: RealArray, z: RealArray, output_log_space: bool = False):
    """Computes Kve(v, z) and Kve(v + 1., z) via Power series expansion."""
    # This is based on:
    # [1] N. Temme, On the Numerical Evaluation of the Modified Bessel Function
    #     of the Third Kind. Journal of Computational Physics 19, 1975.
    # [2] Numerical Recipes in C. The Art of Scientific Computing,
    #     2nd Edition, 1992
    # We will assume |z| <= 2. and |v| < 0.5 for fast convergence.
    dtype = jnp.result_type(v, z, jnp.float32)
    tol = jnp.asarray(np.finfo(dtype).eps, dtype=dtype)

    # The initial series term is defined by 6.7.39 in [2]. We compute
    # related coefficients and quantities.
    coeff1, coeff2, gamma1pv_inv, gamma1mv_inv = _evaluate_temme_coeffs(v)

    z_sq = jnp.square(z)

    logzo2 = jnp.log(z / 2.)
    mu = -v * logzo2
    sinc_v = jnp.where(
            jnp.equal(v, 0.),
            jnp.asarray(1., dtype=dtype),
            jnp.sin(np.pi * v) / (np.pi * v))
    sinhc_mu = jnp.where(
            jnp.equal(mu, 0.),
            jnp.asarray(1., dtype=dtype),
            jnp.sinh(mu) / mu)
    # These are defined in 6.7.17 in [2].
    initial_f = (coeff1 * jnp.cosh(mu) +
                             coeff2 * -logzo2 * sinhc_mu) / sinc_v
    initial_p = 0.5 * jnp.exp(mu) / gamma1pv_inv
    initial_q = 0.5 * jnp.exp(-mu) / gamma1mv_inv
    max_iterations = 1000

    @dataclass
    class Carry:
        should_stop: BooleanArray
        index: RealArray
        f: RealArray
        p: RealArray
        q: RealArray
        coeff: RealArray
        kv_sum: RealArray
        kvp1_sum: RealArray

    def body_fn(c: Carry) -> Carry:
        f = jnp.where(
                c.should_stop,
                c.f,
                (c.index * c.f + c.p + c.q) / (jnp.square(c.index) - jnp.square(v)))
        p = jnp.where(c.should_stop, c.p, c.p / (c.index - v))
        q = jnp.where(c.should_stop, c.q, c.q / (c.index + v))
        h = p - c.index * f
        # c_k = (z ** 2 / 4) ** k / (k!)
        coeff = jnp.where(c.should_stop, c.coeff, c.coeff * z_sq / (4 * c.index))
        kv_sum = jnp.where(c.should_stop, c.kv_sum, c.kv_sum + coeff * f)
        kvp1_sum = jnp.where(c.should_stop, c.kvp1_sum, c.kvp1_sum + coeff * h)
        index = c.index + 1
        should_stop = ((jnp.abs(coeff * f) < jnp.abs(kv_sum) * tol)
                       | (index > max_iterations))

        return Carry(should_stop, index, f, p, q, coeff, kv_sum, kvp1_sum)

    c = lax.while_loop(cond_fun=lambda c: ~jnp.all(c.should_stop),
                       body_fun=body_fn,
                       init_val=Carry(jnp.zeros_like(initial_f, dtype=jnp.bool),
                                      jnp.asarray(1., dtype),
                                      initial_f,
                                      initial_p,
                                      initial_q,
                                      jnp.ones_like(initial_p),
                                      initial_f,
                                      initial_p))

    log_kve = jnp.log(c.kv_sum) + z
    log_kvep1 = jnp.log(2. * c.kvp1_sum) + z - jnp.log(z)
    if output_log_space:
        return log_kve, log_kvep1

    return jnp.exp(log_kve), jnp.exp(log_kvep1)


def _continued_fraction_kv(v: RealArray, z: RealArray, output_log_space: bool = False
                           ) -> tuple[RealArray, RealArray]:
    """Compute Modified Bessels of Second Kind using Hypergeometric functions.

    First define `k_n(z) = (-1)**n U(v + n + 0.5, 2 * v + 1., 2 * z: RealArray)` where
    `U(a, b, z)` is the confluent hypergeometric function.

    We can compute via [1] `K_v(z)` and `K_{v + 1}(z)` via the identities:

    `K_v(z) = sqrt(pi) * (2 * z) ** v * exp(-z) * k_0(z)`,
    `K_{v + 1}(z) = K_v(z) * (v + z + 0.5 - k_1(z) / k_0(z)`,

    This function aims to compute the ratio `k_1(z) / k_0(z)` via
    a continued fraction, under the assumption |v| < 0.5, and finally
    `K_v(z)` and `K_{v + 1}(z)`.

    Args:
        v: Floating-point `Tensor` broadcastable with `z`.
        z: Floating-point `Tensor` broadcastable with `v`.
        output_log_space: `bool`. If `True`, output is in log-space.
            Default value: `False`.

    Returns:
        kv_tuple: `K_v(z)` and `K_{v + 1}(z)`.

    #### References
    [1] N. Temme, On the Numerical Evaluation of the Modified Bessel Function
        of the Third Kind. Journal of Computational Physics 19, 1975.
    [2] J. Campbell. On Temme's Algorithm for the Modified Bessel Function
        of the Third Kind. https://dl.acm.org/doi/pdf/10.1145/355921.355928
    [3] Numerical Recipes in C. The Art of Scientific Computing,
        2nd Edition, 1992
    """
    dtype = jnp.result_type(v, z, jnp.float32)
    tol = jnp.asarray(np.finfo(dtype).eps, dtype=dtype)
    max_iterations = 1000

    @dataclass
    class Carry:
        should_stop: BooleanArray
        index: RealArray
        partial_numerator: RealArray
        partial_denominator: RealArray
        denominator_ratio: RealArray
        convergent_difference: RealArray
        hypergeometric_ratio: RealArray
        # Terms for recurrence in 6.7.36 in [3].
        k_0: RealArray
        k_1: RealArray
        # Intermediate coefficient in 6.7.30 in [3].
        c: RealArray
        # Intermediate sum in 6.7.35 in [3].
        q: RealArray
        hypergeometric_sum: RealArray

    # Use Steed's algorithm to evaluate the confluent hypergeometric
    # function continued fraction in a numerically stable manner.
    def steeds_algorithm(carry: Carry) -> Carry:
        # The numerator is v**2 - (c.index - 0.5) ** 2
        partial_numerator = carry.partial_numerator - 2.0 * (carry.index - 1.0)
        c = jnp.where(carry.should_stop, carry.c,
                      -carry.c * partial_numerator / carry.index)
        next_k = (carry.k_0
                  - carry.partial_denominator * carry.k_1) / partial_numerator
        k_0 = jnp.where(carry.should_stop, carry.k_0, carry.k_1)
        k_1 = jnp.where(carry.should_stop, carry.k_1, next_k)
        q = jnp.where(carry.should_stop, carry.q, carry.q + c * next_k)
        partial_denominator = carry.partial_denominator + 2.0
        denominator_ratio = 1. / (
                partial_denominator + partial_numerator * carry.denominator_ratio)
        convergent_difference = jnp.where(
                carry.should_stop, carry.convergent_difference,
                carry.convergent_difference * (
                        partial_denominator * denominator_ratio - 1.))
        hypergeometric_ratio = jnp.where(
                carry.should_stop,
                carry.hypergeometric_ratio,
                carry.hypergeometric_ratio + convergent_difference)
        hypergeometric_sum = jnp.where(
                carry.should_stop,
                carry.hypergeometric_sum,
                carry.hypergeometric_sum + q * convergent_difference)
        index = carry.index + 1
        should_stop = (jnp.abs(q * convergent_difference) <
                                     jnp.abs(hypergeometric_sum) * tol) | (
                                             index > max_iterations)
        return Carry(should_stop,
                     index,
                     partial_numerator,
                     partial_denominator,
                     denominator_ratio,
                     convergent_difference,
                     hypergeometric_ratio,
                     k_0, k_1, c, q, hypergeometric_sum)

    initial_numerator = jnp.square(v) - 0.25
    initial_denominator = 2 * (z + 1.)
    initial_ratio = 1. / initial_denominator + jnp.zeros_like(v)
    initial_seq = -initial_numerator + jnp.zeros_like(z)

    carry = lax.while_loop(
             cond_fun=lambda c: ~jnp.all(c.should_stop),
             body_fun=steeds_algorithm,
             init_val=Carry(jnp.zeros_like(v + z, dtype=jnp.bool),
                            jnp.asarray(2., dtype=dtype),
                            initial_numerator,
                            initial_denominator,
                            initial_ratio,
                            initial_ratio,
                            initial_ratio,
                            jnp.zeros_like(v + z),
                            jnp.ones_like(v + z),
                            initial_seq,
                            initial_seq,
                            1 - initial_numerator * initial_ratio))
    log_kve = 0.5 * jnp.log(np.pi / (2 * z)) - jnp.log(carry.hypergeometric_sum)
    log_kvp1e = (
            log_kve + jnp.log1p(
                    2 * (v + z + initial_numerator * carry.hypergeometric_ratio))
            - jnp.log(z) - jnp.asarray(np.log(2.), dtype=dtype))
    if output_log_space:
        return log_kve, log_kvp1e
    return jnp.exp(log_kve), jnp.exp(log_kvp1e)


def _temme_expansion(v: RealArray, x: RealArray, output_log_space: bool = False
                     ) -> tuple[RealArray, RealArray]:
    """Compute modified bessel functions using Temme's method."""
    # The implementation of this is based on [1].
    # [1] N. Temme, On the Numerical Evaluation of the Modified Bessel Function
    #     of the Third Kind. Journal of Computational Physics 19, 1975.
    dtype = jnp.result_type(v, x, jnp.float32)
    v_less_than_zero = v < 0.
    v = jnp.abs(v)
    n = jnp.round(v)
    # Use this to compute Kv(u, x) and Kv(u + 1., x)
    u = v - n
    x_abs = jnp.abs(x)

    small_x = jnp.where(x_abs <= 2., x_abs, jnp.asarray(0.1, dtype=dtype))
    large_x = jnp.where(x_abs > 2., x_abs, jnp.asarray(1000., dtype=dtype))
    temme_kue, temme_kuep1 = _temme_series(
            u, small_x, output_log_space=output_log_space)
    cf_kue, cf_kuep1 = _continued_fraction_kv(
            u, large_x, output_log_space=output_log_space)

    kue = jnp.where(x_abs <= 2., temme_kue, cf_kue)
    kuep1 = jnp.where(x_abs <= 2., temme_kuep1, cf_kuep1)

    # Now use the forward recurrence for modified bessel functions
    # to compute Kv(v, x). That is,
    # K_{v + 1}(z) - (2v / z) K_v(z) - K_{v - 1}(z) = 0.
    # This is known to be forward numerically stable.
    # Note: This recurrence is also satisfied by K_v(z) * exp(z)

    @dataclass
    class Carry:
        index: RealArray
        kve: RealArray
        kvep1: RealArray

    def bessel_recurrence(c: Carry) -> Carry:
        if output_log_space:
            next_kvep1 = jnp.logaddexp(
                    c.kvep1 + jnp.log(u + c.index) +
                    jnp.asarray(np.log(2.), dtype=dtype) - jnp.log(x_abs), c.kve)
        else:
            next_kvep1 = 2 * (u + c.index) * c.kvep1 / x_abs + c.kve
        kve = jnp.where(c.index > n, c.kve, c.kvep1)
        kvep1 = jnp.where(c.index > n, c.kvep1, next_kvep1)
        return Carry(c.index + 1., kve, kvep1)

    c = lax.while_loop(
            cond_fun=lambda c: jnp.all(c.index <= n),
            body_fun=bessel_recurrence,
            init_val=Carry(jnp.asarray(1., dtype=dtype), kue, kuep1))
    # Finally, it is known that the Wronskian
    # det(I_v * K'_v - K_v * I'_v) = - 1. / x. We can
    # use this to evaluate I_v by taking advantage of identities of Bessel
    # derivatives.

    if output_log_space:
        ive = -jnp.log(x_abs) - jnp.logaddexp(
                c.kve + jnp.log(bessel_iv_ratio(v + 1., x)), c.kvep1)
    else:
        ive = jnp.reciprocal(
                x_abs * (c.kve * bessel_iv_ratio(v + 1., x) + c.kvep1))

    # We need to add a correction term for negative v.

    if output_log_space:
        log_ive = ive
        negative_v_correction = c.kve - 2. * x_abs
    else:
        log_ive = jnp.log(ive)
        negative_v_correction = jnp.log(c.kve) - 2. * x_abs

    coeff = 2 / np.pi * jnp.sin(np.pi * u)
    coeff = (1. - 2. * jnp.mod(n, 2.)) * coeff

    lse, sign = log_sub_exp(
            log_ive,
            negative_v_correction + jnp.log(jnp.abs(coeff)),
            return_sign=True)
    sign = jnp.where(coeff < 0., sign, 1.)

    log_ive_negative_v = jnp.where(
            coeff < 0.,
            lse,
            jnp.logaddexp(
                    log_ive, negative_v_correction + jnp.log(jnp.abs(coeff))))

    z = u + jnp.mod(n, 2.)

    if output_log_space:
        ive = jnp.where(v_less_than_zero, log_ive_negative_v, ive)

        ive = jnp.where(
                jnp.equal(x, 0.),
                jnp.where(jnp.equal(v, 0.), jnp.asarray(0., dtype=dtype),
                          jnp.asarray(-np.inf, dtype=dtype)), ive)
    else:
        ive = jnp.where(
                v_less_than_zero, sign * jnp.exp(log_ive_negative_v), ive)

        ive = jnp.where(
                jnp.equal(x, 0.),
                jnp.where(jnp.equal(v, 0.), jnp.asarray(1., dtype=dtype),
                          jnp.asarray(0., dtype=dtype)), ive)

    ive = jnp.where(jnp.equal(x, 0.) & v_less_than_zero,
                                 jnp.where(
                                         jnp.equal(z, jnp.floor(z)),
                                         ive,
                                         jnp.asarray(np.inf, dtype=dtype)), ive)

    kve = jnp.where(jnp.equal(x, 0.), jnp.asarray(np.inf, dtype=dtype), c.kve)
    ive = jnp.where(x < 0., jnp.asarray(np.nan, dtype=dtype), ive)
    kve = jnp.where(x < 0., jnp.asarray(np.nan, dtype=dtype), kve)
    return ive, kve


def _bessel_ive_shared(v: RealArray, z: RealArray, output_log_space: bool = False):
    """Compute bessel_ive(v, z)."""
    # I_{-v} == I_{v} for negative integers
    v_is_integer = jnp.equal(jnp.floor(v), v)
    v_abs = jnp.where((v < 0.) & v_is_integer, -v, v)

    z_abs = jnp.abs(z)
    # Handle the zero case specially.
    z_abs = jnp.where(jnp.equal(z_abs, 0.), 1., z_abs)

    small_v = jnp.where(jnp.abs(v_abs) < 50., v_abs, 0.1)
    large_v = jnp.where(jnp.abs(v_abs) >= 50., v_abs, 1000.)

    olver_ive, _ = _olver_asymptotic_uniform(
            large_v, z_abs, output_log_space=output_log_space)
    temme_ive = _temme_expansion(
            small_v, z_abs, output_log_space=output_log_space)[0]
    ive = jnp.where(jnp.abs(v) >= 50., olver_ive, temme_ive)

    # Handle when z is zero.
    if output_log_space:
        ive = jnp.where(
                jnp.equal(z, 0.),
                jnp.where(
                        jnp.equal(v, 0.),
                        0.,
                        jnp.where(
                                v_abs < 0.,
                                jnp.inf,
                                -jnp.inf)), ive)
    else:
        ive = jnp.where(
                jnp.equal(z, 0.),
                jnp.where(
                        jnp.equal(v, 0.),
                        1.,
                        jnp.where(
                                v_abs < 0.,
                                np.inf,
                                0.)), ive)

    # Handle when z < 0.
    ive = jnp.where((z < 0.) & ~v_is_integer, jnp.nan, ive)
    # If v is an odd integer, we flip sign of the computation.
    if not output_log_space:
        ive = jnp.where((z < 0.) & v_is_integer & jnp.not_equal(
                2. * jnp.floor(v / 2.), v), -ive, ive)
    return ive


def _bessel_ive_naive(v: RealArray, z: RealArray) -> RealArray:
    """Compute bessel_ive(v, z)."""
    return _bessel_ive_shared(v, z, output_log_space=False)


# def _bessel_ive_fwd(v: RealArray, z: RealArray):
#     """Compute output, aux (collaborates with _bessel_ive_bwd)."""
#     output = _bessel_ive_naive(v, z)
#     return output, (v, z)
#
#
# def _bessel_ive_bwd(aux: RealArray, g: RealArray):
#     """Reverse mode impl for bessel_ive."""
#     v, z = aux
#     ive = _bessel_ive_custom_gradient(v, z)
#     grad_z = g * (
#             _bessel_ive_custom_gradient(v + 1., z) + (v / z - jnp.sign(z)) * ive)
#     # _, grad_z = tfp_math.fix_gradient_for_broadcasting(
#     #         [v, z], [jnp.ones_like(grad_z), grad_z])
#
#     # No gradient for v at the moment. This is a complicated expression
#     # The gradient with respect to the parameter doesn't have an easy closed
#     # form. More work will need to be done to ensure good numerics for the
#     # gradient.
#     # TODO(b/169357627): Implement gradients of modified bessel functions with
#     # respect to parameters.
#
#     return None, grad_z


def _bessel_ive_jvp(primals: tuple[RealArray, RealArray],
                    tangents: tuple[RealArray, RealArray]
                    ) -> tuple[RealArray, RealArray]:
    """Computes JVP for bessel_ive (supports JAX custom derivative)."""
    v, z = primals
    dv, dz = tangents

    # TODO(https://github.com/google/jax/issues/3768): eliminate broadcast_to?
    bc_shp = jnp.broadcast_shapes(jnp.shape(dv), jnp.shape(dz))
    dz = jnp.broadcast_to(dz, bc_shp)

    ive = _bessel_ive_custom_gradient(v, z)
    pz = _bessel_ive_custom_gradient(v + 1., z) + (v / z - jnp.sign(z)) * ive

    # `bessel_ive` does not have gradients with respect to `v`, and thus
    # this `JVP` rule matches TF.
    # Ideally, it would be nice to throw an exception when taking gradients of
    # in JAX mode, but this is not possible at the moment with `custom_jvp`.
    # See https://github.com/google/jax/issues/5913 for details.
    # TODO(https://github.com/google/jax/issues/5913): Define vjp for v.

    return ive, pz * dz


@custom_jvp
def _bessel_ive_custom_gradient(v: RealArray, z: RealArray) -> RealArray:
    return _bessel_ive_naive(v, z)


_bessel_ive_custom_gradient.defjvp(_bessel_ive_jvp)


def bessel_ive(v: RealArray, z: RealArray):
    """Computes exponentially scaled modified Bessel function of the first kind.

    This function computes `Ive`, which is an exponentially scaled version
    of the modified Bessel function of the first kind.

    `Ive(v, z) = Iv(v, z) * exp(-abs(z))`

    Warning: Gradients with respect to the first parameter `v` are currently not
    defined.

    Args:
        v: Floating-point `Tensor` broadcastable with `z` for which `Ive(v, z)`
            should be computed. `v` is expected to be non-negative.
        z: Floating-point `Tensor` broadcastable with `v` for which `Ive(v, z)`
            should be computed. If `z` is negative, `v` is expected to be an integer.
        name: A name for the operation (optional).
            Default value: `None` (i.e., 'bessel_ive').

    Returns:
        bessel_ive: Exponentially modified Bessel Function of the first kind.
    """
    dtype = jnp.result_type(v, z, jnp.float32)
    v = jnp.asarray(v, dtype=dtype)
    z = jnp.asarray(z, dtype=dtype)
    return _bessel_ive_custom_gradient(v, z)


def _bessel_kve_shared(v: RealArray, z: RealArray, output_log_space: bool = False):
    """Compute bessel_kve(v, z)."""
    dtype = jnp.result_type(v, z, jnp.float32)
    v = jnp.asarray(v, dtype=dtype)
    z = jnp.asarray(z, dtype=dtype)

    # K_{-v} == K_{v} for negative values.
    v = jnp.abs(v)

    z_abs = jnp.abs(z)

    small_v = jnp.where(v < 50., v, 0.1)
    large_v = jnp.where(v >= 50., v, 1000.)

    _, olver_kve = _olver_asymptotic_uniform(
            large_v, z_abs, output_log_space=output_log_space)
    temme_kve = _temme_expansion(
            small_v, z_abs, output_log_space=output_log_space)[1]
    kve = jnp.where(v >= 50., olver_kve, temme_kve)

    # Handle when z is zero.
    kve = jnp.where(jnp.equal(z, 0.), np.inf, kve)
    return jnp.where(z < 0., np.nan, kve)


def _bessel_kve_naive(v: RealArray, z: RealArray) -> RealArray:
    """Compute bessel_kve(v, z)."""
    return _bessel_kve_shared(v, z, output_log_space=False)


# def _bessel_kve_fwd(v: RealArray, z: RealArray
#                     ) -> tuple[RealArray, tuple[RealArray, RealArray]]:
#     """Compute output, aux (collaborates with _bessel_kve_bwd)."""
#     output = _bessel_kve_naive(v, z)
#     return output, (v, z)
#
#
# def _bessel_kve_bwd(aux: RealArray, g: RealArray) -> tuple[None, RealArray]:
#     """Reverse mode impl for bessel_kve."""
#     v, z = aux
#     kve = _bessel_kve_custom_gradient(v, z)
#     grad_z = g * ((z - v) / z * kve - _bessel_kve_custom_gradient(v - 1., z))
#     # _, grad_z = tfp_math.fix_gradient_for_broadcasting(
#     #         [v, z], [jnp.ones_like(grad_z), grad_z])
#
#     # No gradient for v at the moment. This is a complicated expression
#     # The gradient with respect to the parameter doesn't have an easy closed
#     # form. More work will need to be done to ensure good numerics for the
#     # gradient.
#     # TODO(b/169357627): Implement gradients of modified bessel functions with
#     # respect to parameters.
#
#     return None, grad_z


def _bessel_kve_jvp(primals: Any, tangents: Any) -> tuple[RealArray, RealArray]:
    """Computes JVP for bessel_kve (supports JAX custom derivative)."""
    v, z, = primals
    _, dz = tangents

    # TODO(https://github.com/google/jax/issues/3768): eliminate broadcast_to?
    bc_shp = jnp.broadcast_shapes(jnp.shape(v), jnp.shape(dz))
    dz = jnp.broadcast_to(dz, bc_shp)

    kve = _bessel_kve_custom_gradient(v, z)
    pz = (z - v) / z * kve - _bessel_kve_custom_gradient(v - 1., z)

    # `bessel_kve` does not have gradients with respect to `v`, and thus
    # this `JVP` rule matches TF.
    # Ideally, it would be nice to throw an exception when taking gradients of
    # in JAX mode, but this is not possible at the moment with `custom_jvp`.
    # See https://github.com/google/jax/issues/5913 for details.
    # TODO(https://github.com/google/jax/issues/5913): Define vjp for v.

    return kve, pz * dz


@custom_jvp
def _bessel_kve_custom_gradient(v: RealArray, z: RealArray) -> RealArray:
    return _bessel_kve_naive(v, z)


_bessel_kve_custom_gradient.defjvp(_bessel_kve_jvp)


def bessel_kve(v: RealArray, z: RealArray):
    """Computes exponentially scaled modified Bessel function of the 2nd kind.

    This function computes `Kve` which is an exponentially scaled version
    of the modified Bessel function of the first kind.

    `Kve(v, z) = Kv(v, z) * exp(abs(z))`

    Warning: Gradients with respect to the first parameter `v` are currently not
    defined.

    Args:
        v: Floating-point `Tensor` broadcastable with `z` for which `Kve(v, z)`
            should be computed. `v` is expected to be non-negative.
        z: Floating-point `Tensor` broadcastable with `v` for which `Kve(v, z)`
            should be computed. If `z` is negative, `v` is expected to be an integer.
        name: A name for the operation (optional).
            Default value: `None` (i.e., 'bessel_kve').

    Returns:
        bessel_kve: Exponentially modified Bessel Function of the 2nd kind.
    """
    dtype = jnp.result_type(v, z, jnp.float32)
    v = jnp.asarray(v, dtype=dtype)
    z = jnp.asarray(z, dtype=dtype)
    return _bessel_kve_custom_gradient(v, z)


def _log_bessel_ive_naive(v: RealArray, z: RealArray) -> RealArray:
    return _bessel_ive_shared(v, z, output_log_space=True)


# def _log_bessel_ive_fwd(v: RealArray, z: RealArray
#                         ) -> tuple[RealArray, tuple[RealArray, RealArray]]:
#     """Compute output, aux (collaborates with _log_bessel_ive_bwd)."""
#     output = _log_bessel_ive_naive(v, z)
#     return output, (v, z)
#
#
# def _log_bessel_ive_bwd(aux: RealArray, g: RealArray):
#     """Reverse mode impl for log_bessel_ive."""
#     v, z = aux
#     grad_z = g * (bessel_iv_ratio(v + 1., z) + (v / z - jnp.sign(z)))
#     # _, grad_z = tfp_math.fix_gradient_for_broadcasting(
#     #         [v, z], [jnp.ones_like(grad_z), grad_z])
#
#     # No gradient for v at the moment. This is a complicated expression
#     # The gradient with respect to the parameter doesn't have an easy closed
#     # form. More work will need to be done to ensure good numerics for the
#     # gradient.
#     # TODO(b/169357627): Implement gradients of modified bessel functions with
#     # respect to parameters.
#
#     return None, grad_z


def _log_bessel_ive_jvp(primals: Any, tangents: Any) -> tuple[RealArray, RealArray]:
    """Computes JVP for log_bessel_ive (supports JAX custom derivative)."""
    v, z = primals
    _, dz = tangents

    # TODO(https://github.com/google/jax/issues/3768): eliminate broadcast_to?
    bc_shp = jnp.broadcast_shapes(jnp.shape(v), jnp.shape(dz))
    dz = jnp.broadcast_to(dz, bc_shp)

    log_ive = _log_bessel_ive_naive(v, z)

    pz = bessel_iv_ratio(v + 1., z) + (v / z - jnp.sign(z))

    # `log_bessel_ive` does not have gradients with respect to `v`, and thus
    # this `JVP` rule matches TF.
    # Ideally, it would be nice to throw an exception when taking gradients of
    # in JAX mode, but this is not possible at the moment with `custom_jvp`.
    # See https://github.com/google/jax/issues/5913 for details.
    # TODO(https://github.com/google/jax/issues/5913): Define vjp for v.

    return log_ive, pz * dz


@custom_jvp
def _log_bessel_ive_custom_gradient(v: RealArray, z: RealArray) -> RealArray:
    return _log_bessel_ive_naive(v, z)


_log_bessel_ive_custom_gradient.defjvp(_log_bessel_ive_jvp)


def log_bessel_ive(v: RealArray, z: RealArray):
    """Computes `log(tfp.math.bessel_ive(v, z))`.

    This function is a more numerically stable version of
    `log(tfp.math.bessel_ive(v, z))`, along with more numerically stable
    gradients.

    Warning: Gradients with respect to the first parameter `v` are currently not
    defined.

    Args:
        v: Floating-point `Tensor` broadcastable with `z` for which `log(Ive(v, z))`
            should be computed. `v` is expected to be non-negative.
        z: Floating-point `Tensor` broadcastable with `v` for which `log(Ive(v, z))`
            should be computed. If `z` is negative, `v` is expected to be an integer.
        name: A name for the operation (optional).
            Default value: `None` (i.e., 'log_bessel_ive').

    Returns:
        log_bessel_ive: Log of Exponentially modified Bessel Function of the first
            kind.
    """
    dtype = jnp.result_type(v, z, jnp.float32)
    v = jnp.asarray(v, dtype=dtype)
    z = jnp.asarray(z, dtype=dtype)
    return _log_bessel_ive_custom_gradient(v, z)


def _log_bessel_kve_naive(v: RealArray, z: RealArray) -> RealArray:
    """Compute log(bessel_kve(v, z))_."""
    return _bessel_kve_shared(v, z, output_log_space=True)


# def _log_bessel_kve_fwd(v: RealArray, z: RealArray
#                         ) -> tuple[RealArray, tuple[RealArray, RealArray]]:
#     """Compute output, aux (collaborates with _log_bessel_kve_bwd)."""
#     output = _log_bessel_kve_naive(v, z)
#     return output, (v, z)


# def _log_bessel_kve_bwd(aux: RealArray, g: RealArray) -> tuple[RealArray, RealArray]:
#     """Reverse mode impl for bessel_kve."""
#     v, z = aux
#     log_kve = _log_bessel_kve_custom_gradient(v, z)
#     grad_z = jnp.logaddexp(
#             _log_bessel_kve_custom_gradient(v - 1., z),
#             _log_bessel_kve_custom_gradient(v + 1., z)) - np.log(2.) - log_kve
#     grad_z = g * -jnp.expm1(grad_z)
#     # _, grad_z = tfp_math.fix_gradient_for_broadcasting(
#     #         [v, z], [jnp.ones_like(grad_z), grad_z])
#
#     # No gradient for v at the moment. This is a complicated expression
#     # The gradient with respect to the parameter doesn't have an easy closed
#     # form. More work will need to be done to ensure good numerics for the
#     # gradient.
#     # TODO(b/169357627): Implement gradients of modified bessel functions with
#     # respect to parameters.
#
#     return None, grad_z


def _log_bessel_kve_jvp(primals: list[RealArray], tangents: list[RealArray]):
    """Computes JVP for bessel_kve (supports JAX custom derivative)."""
    v, z = primals
    _, dz = tangents

    # TODO(https://github.com/google/jax/issues/3768): eliminate broadcast_to?
    bc_shp = jnp.broadcast_shapes(jnp.shape(v), jnp.shape(dz))
    dz = jnp.broadcast_to(dz, bc_shp)

    log_kve = _log_bessel_kve_custom_gradient(v, z)
    pz = jnp.logaddexp(
            _log_bessel_kve_custom_gradient(v - 1., z),
            _log_bessel_kve_custom_gradient(v + 1., z)) - np.log(2.) - log_kve
    pz = -jnp.expm1(pz)

    # `bessel_kve` does not have gradients with respect to `v`, and thus
    # this `JVP` rule matches TF.
    # Ideally, it would be nice to throw an exception when taking gradients of
    # in JAX mode, but this is not possible at the moment with `custom_jvp`.
    # See https://github.com/google/jax/issues/5913 for details.
    # TODO(https://github.com/google/jax/issues/5913): Define vjp for v.

    return log_kve, pz * dz


@custom_jvp
def _log_bessel_kve_custom_gradient(v: RealArray, z: RealArray) -> RealArray:
    return _log_bessel_kve_naive(v, z)


_log_bessel_kve_custom_gradient.defjvp(_log_bessel_kve_jvp)
# vjp_fwd=_log_bessel_kve_fwd,
# vjp_bwd=_log_bessel_kve_bwd,


def log_bessel_kve(v: RealArray, z: RealArray) -> RealArray:
    """Computes `log(tfp.math.bessel_kve(v, z))`.

    This function is a more numerically stable version of
    `log(tfp.math.bessel_kve(v, z))`.

    Warning: Gradients with respect to the first parameter `v` are currently not
    defined.

    Args:
        v: Floating-point `Tensor` broadcastable with `z` for which `log(Kve(v, z))`
            should be computed. `v` is expected to be non-negative.
        z: Floating-point `Tensor` broadcastable with `v` for which `log(Kve(v, z))`
            should be computed. If `z` is negative, `v` is expected to be an integer.
        name: A name for the operation (optional).
            Default value: `None` (i.e., 'log_bessel_kve').

    Returns:
        log_bessel_kve: Log of Exponentially modified Bessel Function of the second
            kind.
    """
    dtype = jnp.result_type(v, z, jnp.float32)
    v = jnp.asarray(v, dtype=dtype)
    z = jnp.asarray(z, dtype=dtype)
    return _log_bessel_kve_custom_gradient(v, z)
