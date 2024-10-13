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
"""Tests for special."""

import functools
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax import Array, jit, tree
from jax.random import key
from jax.typing import DTypeLike
from scipy import constants as scipy_constants
from scipy import special as scipy_special

from tjax import RngStream, assert_tree_allclose, bessel
from tjax.bessel import _compute_general_continued_fraction

DType = np.dtype[Any]


def _compute_numerical_jacobian(f, xs, i, scale=1e-3):
    """Compute the numerical jacobian of `f`."""
    dtype_i = xs[i].dtype
    shape_i = xs[i].shape
    size_i = np.prod(shape_i, dtype=np.int32)

    def grad_i(d):
        return (f(*(xs[:i] + [xs[i] + d * scale] + xs[i + 1:]))
                        - f(*(xs[:i] + [xs[i] - d * scale] + xs[i + 1:]))) / (2. * scale)

    ret = jax.vmap(grad_i)(
            np.eye(size_i, dtype=dtype_i).reshape((size_i, *shape_i)))

    ret = jnp.transpose(ret, jnp.roll(jnp.arange(len(ret.shape)), -1))
    return jnp.reshape(ret, ret.shape[:-1] + shape_i)


def compute_max_gradient_error(f, xs, scale=1e-3, eval_fn=None):
    """Compute the max difference between autodiff and numerical jacobian."""
    xs = tree.map(jnp.asarray, xs)
    f_jac = jax.jacrev(f, argnums=range(len(xs)))
    theoretical_jacobian = f_jac(*xs)

    numerical_jacobian = [_compute_numerical_jacobian(f, xs, i, scale)
                          for i in range(len(xs))]
    theoretical_jacobian, numerical_jacobian = tree.map(
            jnp.asarray,
            [theoretical_jacobian, numerical_jacobian])
    return np.max(np.array(
            [np.max(np.abs(a - b))
             for (a, b) in zip(theoretical_jacobian, numerical_jacobian, strict=False)]))


def tf_uniform(shape: list[int],
               minval: float = 0.0,
               maxval: float = 1.0,
               *,
               dtype: DTypeLike = float,
               seed: Any
               ) -> Array:
    return jr.uniform(seed, tuple(shape), minval=minval, maxval=maxval, dtype=dtype)


def test_continued_fraction() -> None:
    # Check that the simplest continued fraction returns the golden ratio.
    assert_tree_allclose(
            (
                    _compute_general_continued_fraction(
                            100, [], partial_numerator_fn=lambda _: 1.)),
            scipy_constants.golden - 1.)

    # Check the continued fraction constant is returned.
    cf_constant_denominators = scipy_special.i1(2.) / scipy_special.i0(2.)

    assert_tree_allclose(
            (
                    _compute_general_continued_fraction(
                            100,
                            [],
                            partial_denominator_fn=lambda i: i,
                            tolerance=1e-5)),
            cf_constant_denominators, rtol=1e-5)

    cf_constant_numerators = np.sqrt(2 / (np.e * np.pi)) / (
            scipy_special.erfc(np.sqrt(0.5))) - 1.

    # Check that we can specify dtype and tolerance.
    assert_tree_allclose(
            (
                    _compute_general_continued_fraction(
                            100, [], partial_numerator_fn=lambda i: i,
                            tolerance=1e-5,
                            dtype=jnp.float64)),
            cf_constant_numerators, rtol=1e-5)


def verify_bessel_iv_ratio(v: Array, z: Array, rtol: float) -> None:
    bessel_iv_ratio, v, z = ([bessel.bessel_iv_ratio(v, z), v, z])
    # Use ive to avoid nans.
    scipy_ratio = scipy_special.ive(v, z) / scipy_special.ive(v - 1., z)
    assert_tree_allclose(bessel_iv_ratio, scipy_ratio, rtol=rtol)


def test_bessel_iv_ratio_v_and_z_small() -> None:
    seed_stream = RngStream(key(0))
    z = tf_uniform([int(1e3)], seed=seed_stream.key())
    v = tf_uniform([int(1e3)], seed=seed_stream.key())
    # When both values are small, both the scipy ratio and
    # the computation become numerically unstable.
    # Anecdotally (when comparing to mpmath) the computation is more often
    # 'right' compared to the naive ratio.

    bessel_iv_ratio, v, z = ([bessel.bessel_iv_ratio(v, z), v, z])
    scipy_ratio = scipy_special.ive(v, z) / scipy_special.ive(v - 1., z)

    safe_scipy_values = np.where(
            ~np.isnan(scipy_ratio) & (scipy_ratio != 0.))

    assert_tree_allclose(
            bessel_iv_ratio[safe_scipy_values],
            scipy_ratio[safe_scipy_values], rtol=3e-4, atol=1e-6)


def test_bessel_iv_ratio_v_and_z_medium() -> None:
    seed_stream = RngStream(key(0))
    z = tf_uniform([int(1e3)], 1., 10., seed=seed_stream.key())
    v = tf_uniform([int(1e3)], 1., 10., seed=seed_stream.key())
    verify_bessel_iv_ratio(v, z, rtol=7e-6)


def test_bessel_iv_ratio_v_and_z_large() -> None:
    seed_stream = RngStream(key(0))
    # Use 50 as a cap. It's been observed that for v > 50, that
    # the scipy ratio can be quite wrong compared to mpmath.
    z = tf_uniform([int(1e3)], 10., 50., seed=seed_stream.key())
    v = tf_uniform([int(1e3)], 10., 50., seed=seed_stream.key())

    # For large v, z, scipy can return NaN values. Filter those out.
    bessel_iv_ratio, v, z = ([bessel.bessel_iv_ratio(v, z), v, z])
    # Use ive to avoid nans.
    scipy_ratio = scipy_special.ive(v, z) / scipy_special.ive(v - 1., z)
    # Exclude zeros and NaN's from scipy. This can happen because the
    # individual function computations may zero out, and thus cause issues
    # in the ratio.
    safe_scipy_values = np.where(
            ~np.isnan(scipy_ratio) & (scipy_ratio != 0.))

    assert_tree_allclose(
            bessel_iv_ratio[safe_scipy_values],
            scipy_ratio[safe_scipy_values],
            # We need to set a high rtol as the scipy computation is numerically
            # unstable.
            rtol=1e-6)


def test_bessel_iv_ratio_v_less_than_z() -> None:
    seed_stream = RngStream(key(0))
    z = tf_uniform([int(1e3)], 1., 10., seed=seed_stream.key())
    # Make v randomly less than z
    v = z * tf_uniform([int(1e3)], 0.1, 0.5, seed=seed_stream.key())
    verify_bessel_iv_ratio(v, z, rtol=6e-6)


def test_bessel_iv_ratio_v_greater_than_z() -> None:
    seed_stream = RngStream(key(0))
    v = tf_uniform([int(1e3)], 1., 10., seed=seed_stream.key())
    # Make z randomly less than v
    z = v * tf_uniform([int(1e3)], 0.1, 0.5, seed=seed_stream.key())
    verify_bessel_iv_ratio(v, z, rtol=1e-6)


def test_bessel_iv_ratio_gradient() -> None:
    v = jnp.asarray([0.5, 1., 10., 20.])[..., jnp.newaxis]
    x = jnp.asarray([0.1, 0.5, 0.9, 1., 12., 14., 22.])

    err = compute_max_gradient_error(
            functools.partial(bessel.bessel_iv_ratio, v), [x])
    assert err < 0.0002


def verify_bessel_ive(v: Array, z: Array, rtol: float, atol: float = 1e-7) -> None:
    bessel_ive, v, z = ([bessel.bessel_ive(v, z), v, z])
    scipy_ive = scipy_special.ive(v, z)
    assert_tree_allclose(bessel_ive, scipy_ive, rtol=rtol, atol=atol)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_bessel_ive_at_zero(dtype: DType) -> None:
    # Check that z = 0 returns 1 for v = 0 and 0 otherwise.
    seed_stream = RngStream(key(0))
    v = tf_uniform([10], 1., 10., seed=seed_stream.key(), dtype=dtype)
    z = jnp.asarray(0., dtype=dtype)
    assert_tree_allclose(
            (bessel.bessel_ive(v, z)), np.zeros([10], dtype=dtype))

    v = jnp.asarray([0.], dtype=dtype)
    assert_tree_allclose(
            (bessel.bessel_ive(v, z)), np.ones([1], dtype=dtype))


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_bessel_ive_z_negative_na_n(dtype: DType) -> None:
    # Check that z < 0 returns NaN for non-integer v.
    seed_stream = RngStream(key(0))
    v = np.linspace(1.1, 10.2, num=11, dtype=dtype)
    z = tf_uniform([11], -10., -1., seed=seed_stream.key(), dtype=dtype)
    bessel_ive = (bessel.bessel_ive(v, z))
    assert np.all(np.isnan(bessel_ive))


@pytest.mark.parametrize(('dtype', 'rtol'),
                         [(np.float32, 1e-6), (np.float64, 1e-6)])
def test_bessel_ive_z_negative_v_integer(dtype: DType, rtol: float) -> None:
    seed_stream = RngStream(key(0))
    v = np.linspace(1., 10., num=10, dtype=dtype)
    z = tf_uniform([10], -10., -1., seed=seed_stream.key(), dtype=dtype)
    z, bessel_ive = ([z, bessel.bessel_ive(v, z)])
    scipy_ive = scipy_special.ive(v, z)
    assert_tree_allclose(bessel_ive, scipy_ive, rtol=rtol)


@pytest.mark.parametrize(('dtype', 'rtol'),
                         [(np.float32, 1e-6), (np.float64, 1e-6)])
def test_bessel_ive_z_negative_v_large(dtype: DType, rtol: float) -> None:
    seed_stream = RngStream(key(0))
    v = np.linspace(100., 200., num=10, dtype=dtype)
    z = tf_uniform([10], -10., -1., seed=seed_stream.key(), dtype=dtype)
    z, bessel_ive = ([z, bessel.bessel_ive(v, z)])
    scipy_ive = scipy_special.ive(v, z)
    assert_tree_allclose(bessel_ive, scipy_ive, rtol=rtol)


@pytest.mark.parametrize(('dtype', 'rtol'),
                         [(np.float32, 1.5e-6), (np.float64, 1e-6)])
def test_bessel_ive_v_and_z_small(dtype: DType, rtol: float) -> None:
    seed_stream = RngStream(key(0))
    z = tf_uniform([int(1e3)], seed=seed_stream.key(), dtype=dtype)
    v = tf_uniform([int(1e3)], seed=seed_stream.key(), dtype=dtype)
    verify_bessel_ive(v, z, rtol=rtol)


@pytest.mark.parametrize(('dtype', 'rtol'),
                         [(np.float32, 3e-6), (np.float64, 1e-6)])
def test_bessel_ive_z_tiny(dtype: DType, rtol: float) -> None:
    seed_stream = RngStream(key(0))
    z = tf_uniform(
            [int(1e3)], 1e-13, 1e-6, seed=seed_stream.key(), dtype=dtype)
    v = tf_uniform([int(1e3)], 0., 10., seed=seed_stream.key(), dtype=dtype)
    verify_bessel_ive(v, z, rtol=rtol, atol=1e-7)


@pytest.mark.parametrize(('dtype', 'rtol'),
                         [(np.float32, 7e-6), (np.float64, 6e-6)])
def test_bessel_ive_v_and_z_medium(dtype: DType, rtol: float) -> None:
    seed_stream = RngStream(key(0))
    z = tf_uniform([int(1e3)], 1., 10., seed=seed_stream.key(), dtype=dtype)
    v = tf_uniform([int(1e3)], 1., 10., seed=seed_stream.key(), dtype=dtype)
    verify_bessel_ive(v, z, rtol=rtol)


@pytest.mark.parametrize(('dtype', 'rtol'),
                         [(np.float32, 1e-6), (np.float64, 1e-6)])
def test_bessel_ive_v_and_z_large(dtype: DType, rtol: float) -> None:
    seed_stream = RngStream(key(0))
    z = tf_uniform([int(1e3)], 10., 50., seed=seed_stream.key(), dtype=dtype)
    v = tf_uniform([int(1e3)], 10., 50., seed=seed_stream.key(), dtype=dtype)
    verify_bessel_ive(v, z, rtol=rtol)


@pytest.mark.parametrize(('dtype', 'rtol'),
                         [(np.float32, 1e-6), (np.float64, 1e-6)])
def test_bessel_ive_v_and_z_very_large(dtype: DType, rtol: float) -> None:
    seed_stream = RngStream(key(0))
    z = tf_uniform(
            [int(1e3)], 50., 100., seed=seed_stream.key(), dtype=dtype)
    v = tf_uniform(
            [int(1e3)], 50., 100., seed=seed_stream.key(), dtype=dtype)
    verify_bessel_ive(v, z, rtol=rtol)


@pytest.mark.parametrize(('dtype', 'rtol'),
                         [(np.float32, 7e-6), (np.float64, 7e-6)])
def test_bessel_ive_v_less_than_z(dtype: DType, rtol: float) -> None:
    seed_stream = RngStream(key(0))
    z = tf_uniform([int(1e3)], 1., 10., seed=seed_stream.key(), dtype=dtype)
    # Make v randomly less than z
    v = z * tf_uniform(
            [int(1e3)], 0.1, 0.5, seed=seed_stream.key(), dtype=dtype)
    verify_bessel_ive(v, z, rtol=rtol)


@pytest.mark.parametrize(('dtype', 'rtol'),
                         [(np.float32, 1e-6), (np.float64, 1e-6)])
def test_bessel_ive_v_greater_than_z(dtype: DType, rtol: float) -> None:
    seed_stream = RngStream(key(0))
    v = tf_uniform([int(1e3)], 1., 10., seed=seed_stream.key(), dtype=dtype)
    # Make z randomly less than v
    z = v * tf_uniform(
            [int(1e3)], 0.1, 0.5, seed=seed_stream.key(), dtype=dtype)
    verify_bessel_ive(v, z, rtol=rtol)


@pytest.mark.parametrize(('dtype', 'rtol'),
                         [(np.float32, 1e-4), (np.float64, 7e-6)])
def test_bessel_ive_v_negative(dtype: DType, rtol: float) -> None:
    seed_stream = RngStream(key(0))
    v = tf_uniform(
            [int(1e3)], -10., -1., seed=seed_stream.key(), dtype=dtype)
    z = tf_uniform([int(1e3)], 1., 15., seed=seed_stream.key(), dtype=dtype)
    verify_bessel_ive(v, z, rtol=rtol)


@pytest.mark.parametrize(('dtype', 'rtol'),
                         [(np.float32, 1e-6), (np.float64, 1e-6)])
def test_bessel_ive_v_zero(dtype: DType, rtol: float) -> None:
    seed_stream = RngStream(key(0))
    v = jnp.asarray(0., dtype=dtype)
    z = tf_uniform([int(1e3)], 1., 15., seed=seed_stream.key(), dtype=dtype)
    verify_bessel_ive(v, z, rtol=rtol)


@pytest.mark.parametrize(('dtype', 'rtol'),
                         [(np.float32, 1e-6), (np.float64, 1e-6)])
def test_bessel_ive_large_z(dtype: DType, rtol: float) -> None:
    seed_stream = RngStream(key(0))
    v = tf_uniform(
            [int(1e3)], minval=0., maxval=0.5, seed=seed_stream.key(), dtype=dtype)
    z = tf_uniform(
            [int(1e3)], minval=100., maxval=10000., seed=seed_stream.key(), dtype=dtype)
    verify_bessel_ive(v, z, rtol=rtol)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_bessel_ive_gradient(dtype: DType) -> None:
    v = jnp.asarray([-1., 0.5, 1., 10., 20.], dtype=dtype)[..., jnp.newaxis]
    z = jnp.asarray([0.2, 0.5, 0.9, 1., 12., 14., 22.], dtype=dtype)

    err = compute_max_gradient_error(
            functools.partial(bessel.bessel_ive, v), [z])
    assert err < 0.0002


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_bessel_ive_negative_gradient(dtype: DType) -> None:
    v = jnp.asarray([1., 10., 20.], dtype=dtype)[..., jnp.newaxis]
    z = jnp.asarray([-.2, -2.5, -3.5, -5.], dtype=dtype)

    err = compute_max_gradient_error(
            functools.partial(bessel.bessel_ive, v), [z])
    assert err < 0.0002


@pytest.mark.parametrize(('dtype', 'rtol'),
                         [(np.float32, 1e-6), (np.float64, 1e-6)])
def test_log_bessel_ive_correct(dtype: DType, rtol: float) -> None:
    seed_stream = RngStream(key(0))
    v = tf_uniform(
            [int(1e3)], minval=0.1, maxval=0.5, seed=seed_stream.key(), dtype=dtype)
    z = tf_uniform(
            [int(1e3)], minval=1., maxval=10., seed=seed_stream.key(), dtype=dtype)
    _, _, log_bessel_ive_expected_, log_bessel_ive_actual_ = ([
            v, z,
            jnp.log(bessel.bessel_ive(v, z)),
            bessel.log_bessel_ive(v, z)
    ])
    assert_tree_allclose(
            log_bessel_ive_expected_, log_bessel_ive_actual_, rtol=rtol)


def test_log_bessel_ive_test_non_inf() -> None:
    # Test that log_bessel_ive(v, z) has more resolution than simply computing
    # log(bessel_ive(v, z)). The inputs below will return -inf in naive float64
    # computation.
    v = np.array([10., 12., 30., 50.], np.float32)
    z = np.logspace(-10., -1., 20).reshape((20, 1)).astype(np.float32)
    assert jnp.all(jnp.isfinite(bessel.log_bessel_ive(v, z)))


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_log_bessel_ive_gradient(dtype: DType, tol: float) -> None:
    v = jnp.asarray([-0.2, -1., 1., 0.5, 2.], dtype=dtype)[..., jnp.newaxis]
    z = jnp.asarray([0.3, 0.5, 0.9, 1., 12., 22.], dtype=dtype)

    err = compute_max_gradient_error(
            functools.partial(bessel.log_bessel_ive, v), [z])
    assert err < tol


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_jit_grad_bcast_log_bessel_ive(dtype: DType) -> None:
    @jit
    def f(v: Array, z: Array):
        dy = jr.normal(key(0), z.shape, dtype=dtype)
        return jax.tree.map(
                lambda t: () if t is None else t,    # session.run doesn't like `None`.
                jax.value_and_grad(
                        lambda v, z: bessel.log_bessel_ive(v, z)**2, (v, z),
                        output_gradients=dy))

    v = jnp.asarray(0.5, dtype=dtype)
    z = jnp.asarray([[0.3, 0.5, 0.9], [1., 12., 22.]], dtype=dtype)

    (f(v, z))


def verify_bessel_kve(v: Array, z: Array, rtol: float) -> None:
    bessel_kve, v, z = ([bessel.bessel_kve(v, z), v, z])
    scipy_kve = scipy_special.kve(v, z)
    assert_tree_allclose(bessel_kve, scipy_kve, rtol=rtol)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_bessel_kve_at_zero(dtype: DType) -> None:
    # Check that z = 0 returns inf for v = 0.
    seed_stream = RngStream(key(0))
    v = tf_uniform([10], 1., 10., seed=seed_stream.key(), dtype=dtype)
    z = jnp.asarray(0., dtype=dtype)
    assert_tree_allclose(
            (bessel.bessel_kve(v, z)),
            np.full([10], np.inf, dtype=dtype))

    v = jnp.asarray([0.], dtype=dtype)
    assert_tree_allclose(
            (bessel.bessel_kve(v, z)), np.full([1],
                                                                                                            np.inf,
                                                                                                            dtype=dtype))


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_bessel_kve_z_negative_na_n(dtype: DType) -> None:
    # Check that z < 0 returns NaN for non-integer v.
    seed_stream = RngStream(key(0))
    v = np.linspace(1.1, 10.2, num=11, dtype=dtype)
    z = tf_uniform([11], -10., -1., seed=seed_stream.key(), dtype=dtype)
    bessel_kve = (bessel.bessel_kve(v, z))
    assert np.all(np.isnan(bessel_kve))


@pytest.mark.parametrize(('dtype', 'rtol'),
                         [(np.float32, 1e-6), (np.float64, 1e-6)])
def test_bessel_kve_z_negative_v_integer(dtype: DType, rtol: float) -> None:
    seed_stream = RngStream(key(0))
    v = np.linspace(1., 10., num=10, dtype=dtype)
    z = tf_uniform([10], -10., -1., seed=seed_stream.key(), dtype=dtype)
    z, bessel_kve = ([z, bessel.bessel_kve(v, z)])
    scipy_kve = scipy_special.kve(v, z)
    assert_tree_allclose(bessel_kve, scipy_kve, rtol=rtol)


@pytest.mark.parametrize(('dtype', 'rtol'),
                         [(np.float32, 1e-6), (np.float64, 1e-6)])
def test_bessel_kve_z_negative_v_large(dtype: DType, rtol: float) -> None:
    seed_stream = RngStream(key(0))
    v = np.linspace(100., 200., num=10, dtype=dtype)
    z = tf_uniform([10], -10., -1., seed=seed_stream.key(), dtype=dtype)
    z, bessel_kve = ([z, bessel.bessel_kve(v, z)])
    scipy_kve = scipy_special.kve(v, z)
    assert_tree_allclose(bessel_kve, scipy_kve, rtol=rtol)


@pytest.mark.parametrize(('dtype', 'rtol'),
                         [(np.float32, 1.5e-6), (np.float64, 1e-6)])
def test_bessel_kve_v_and_z_small(dtype: DType, rtol: float) -> None:
    seed_stream = RngStream(key(0))
    z = tf_uniform([int(1e3)], seed=seed_stream.key(), dtype=dtype)
    v = tf_uniform([int(1e3)], seed=seed_stream.key(), dtype=dtype)
    verify_bessel_kve(v, z, rtol=rtol)


@pytest.mark.parametrize(('dtype', 'rtol'),
                         [(np.float32, 4e-6), (np.float64, 1e-6)])
def test_bessel_kve_v_and_z_medium(dtype: DType, rtol: float) -> None:
    seed_stream = RngStream(key(0))
    z = tf_uniform([int(1e3)], 1., 10., seed=seed_stream.key(), dtype=dtype)
    v = tf_uniform([int(1e3)], 1., 10., seed=seed_stream.key(), dtype=dtype)
    verify_bessel_kve(v, z, rtol=rtol)


@pytest.mark.parametrize(('dtype', 'rtol'),
                         [(np.float32, 3e-6), (np.float64, 1e-6)])
def test_bessel_kve_v_and_z_large(dtype: DType, rtol: float) -> None:
    seed_stream = RngStream(key(0))
    z = tf_uniform([int(1e3)], 10., 50., seed=seed_stream.key(), dtype=dtype)
    v = tf_uniform([int(1e3)], 10., 50., seed=seed_stream.key(), dtype=dtype)
    verify_bessel_kve(v, z, rtol=rtol)


@pytest.mark.parametrize(('dtype', 'rtol'),
                         [(np.float32, 7e-6), (np.float64, 7e-6)])
def test_bessel_kve_v_less_than_z(dtype: DType, rtol: float) -> None:
    seed_stream = RngStream(key(0))
    z = tf_uniform([int(1e3)], 1., 10., seed=seed_stream.key(), dtype=dtype)
    # Make v randomly less than z
    v = z * tf_uniform(
            [int(1e3)], 0.1, 0.5, seed=seed_stream.key(), dtype=dtype)
    verify_bessel_kve(v, z, rtol=rtol)


@pytest.mark.parametrize(('dtype', 'rtol'),
                         [(np.float32, 2e-6), (np.float64, 1e-6)])
def test_bessel_kve_v_greater_than_z(dtype: DType, rtol: float) -> None:
    seed_stream = RngStream(key(0))
    v = tf_uniform([int(1e3)], 1., 10., seed=seed_stream.key(), dtype=dtype)
    # Make z randomly less than v
    z = v * tf_uniform(
            [int(1e3)], 0.1, 0.5, seed=seed_stream.key(), dtype=dtype)
    verify_bessel_kve(v, z, rtol=rtol)


@pytest.mark.parametrize(('dtype', 'rtol'),
                         [(np.float32, 4e-6), (np.float64, 1e-6)])
def test_bessel_kve_v_negative(dtype: DType, rtol: float) -> None:
    seed_stream = RngStream(key(0))
    v = tf_uniform(
            [int(1e3)], -10., -1., seed=seed_stream.key(), dtype=dtype)
    z = tf_uniform([int(1e3)], 1., 15., seed=seed_stream.key(), dtype=dtype)
    verify_bessel_kve(v, z, rtol=rtol)


@pytest.mark.parametrize(('dtype', 'rtol'),
                         [(np.float32, 1e-6), (np.float64, 1e-6)])
def test_bessel_kve_large_z(dtype: DType, rtol: float) -> None:
    seed_stream = RngStream(key(0))
    v = tf_uniform(
            [int(1e3)], minval=0., maxval=0.5, seed=seed_stream.key(), dtype=dtype)
    z = tf_uniform(
            [int(1e3)], minval=100., maxval=10000., seed=seed_stream.key(), dtype=dtype)
    verify_bessel_kve(v, z, rtol=rtol)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_bessel_kve_gradient(dtype: DType) -> None:
    v = jnp.asarray([0.5, 1., 2., 5.])[..., jnp.newaxis]
    z = jnp.asarray([10., 20., 30., 50., 12.])

    err = compute_max_gradient_error(
            functools.partial(bessel.bessel_kve, v), [z])
    assert err < 0.0003


@pytest.mark.parametrize(('dtype', 'rtol', 'atol'),
                         [(np.float32, 1e-6, 1e-5), (np.float64, 1e-6, 1e-6)])
def test_log_bessel_kve_correct(dtype: DType, rtol: float, atol: float
                                ) -> None:
    seed_stream = RngStream(key(0))
    v = tf_uniform(
            [int(1e3)], minval=0.1, maxval=0.5, seed=seed_stream.key(), dtype=dtype)
    z = tf_uniform(
            [int(1e3)], minval=1., maxval=10., seed=seed_stream.key(), dtype=dtype)
    _, _, log_bessel_kve_expected_, log_bessel_kve_actual_ = ([
            v, z,
            jnp.log(bessel.bessel_kve(v, z)),
            bessel.log_bessel_kve(v, z)
    ])
    assert_tree_allclose(
            log_bessel_kve_expected_, log_bessel_kve_actual_, rtol=rtol, atol=atol)


def test_log_bessel_test_non_inf() -> None:
    # Test that log_bessel_kve(v, z) has more resolution than simply computing
    # log(bessel_ive(v, z)). The inputs below will return inf in naive float64
    # computation.
    v = np.array([10., 12., 30., 50.], np.float32)
    z = np.logspace(-10., -1., 20).reshape((20, 1)).astype(np.float32)
    assert jnp.all(jnp.isfinite(bessel.log_bessel_kve(v, z)))


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_log_bessel_kve_gradient(dtype: DType, tol: float) -> None:
    v = jnp.asarray([-0.2, -1., 1., 0.5, 2.], dtype=dtype)[..., jnp.newaxis]
    z = jnp.asarray([0.3, 0.5, 0.9, 1., 12., 22.], dtype=dtype)

    err = compute_max_gradient_error(
            functools.partial(bessel.log_bessel_kve, v), [z])
    assert err < tol
