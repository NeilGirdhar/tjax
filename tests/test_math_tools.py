import math
from functools import partial

import jax
import jax.numpy as jnp
import pytest
import scipy.special as sc
from jax import vjp
from numpy.testing import assert_allclose

from tjax import Array, JaxRealArray, bessel_iv_ratio, divide_where, log_bessel_ive, normalize


@pytest.mark.parametrize(
    ("x", "axis", "result"),
    [
        ((1.0, 2.0, 3.0, 4.0), 0, (0.1, 0.2, 0.3, 0.4)),
        ((0.0, 0.0), 0, (0.5, 0.5)),
    ],
)
def test_l1(
    x: object,
    axis: tuple[int, ...] | int | None,
    result: JaxRealArray,
) -> None:
    assert_allclose(normalize("l1", jnp.asarray(x), axis=axis), jnp.asarray(result))


@pytest.mark.parametrize(
    ("x", "axis", "result"),
    [
        ((5.0, 12.0), 0, (5 / 13, 12 / 13)),
        ((0.0, 0.0), 0, jnp.ones(2) / jnp.sqrt(2)),
    ],
)
def test_l2(
    x: object,
    axis: tuple[int, ...] | int | None,
    result: JaxRealArray,
) -> None:
    assert_allclose(normalize("l2", jnp.asarray(x), axis=axis), jnp.asarray(result))


@pytest.mark.parametrize("k", [0.0, 1.0])
def test_divide_where(k: float) -> None:
    s = (5, 3)
    w = jnp.ones(s) * k
    x = jnp.arange(math.prod(w.shape), dtype="f").reshape(w.shape)
    dummy = jnp.ones_like(w[..., 0])

    def f(w: Array, x: Array, dummy: Array) -> Array:
        total_w = jnp.sum(w, axis=-1)
        return divide_where(
            dividend=jnp.sum(w * x, axis=-1),  # type: ignore
            divisor=total_w,
            where=total_w > 0.0,
            otherwise=dummy,
        )

    y, vjp_f = vjp(partial(f, dummy=dummy), w, x)
    w_bar, x_bar = vjp_f(jnp.ones_like(y))
    assert_allclose(w_bar, k / s[-1] * jnp.tile(jnp.asarray([-1, 0, 1]), (*s[:-1], 1)), atol=1e-3)
    assert_allclose(x_bar, k / s[-1] * jnp.ones(s))


@pytest.mark.parametrize(
    ("v", "x"),
    [(0.0, 0.1), (0.5, 2.0), (2.3, 3.0), (10.0, 5.0)],
)
def test_bessel_iv_ratio_value(v: float, x: float) -> None:
    expected = sc.ive(v, x) / sc.ive(v - 1.0, x)
    assert_allclose(bessel_iv_ratio(jnp.asarray(v), jnp.asarray(x)), expected, rtol=1e-6)


@pytest.mark.parametrize(
    ("v", "x"),
    [(0.5, 0.8), (2.3, 1.5)],
)
def test_bessel_iv_ratio_derivatives(v: float, x: float) -> None:
    step = 1e-4
    expected_dv = (
        sc.ive(v + step, x) / sc.ive(v + step - 1.0, x)
        - sc.ive(v - step, x) / sc.ive(v - step - 1.0, x)
    ) / (2.0 * step)
    expected_dx = (
        sc.ive(v, x + step) / sc.ive(v - 1.0, x + step)
        - sc.ive(v, x - step) / sc.ive(v - 1.0, x - step)
    ) / (2.0 * step)
    actual_dv, actual_dx = jax.jacfwd(bessel_iv_ratio, argnums=(0, 1))(
        jnp.asarray(v), jnp.asarray(x)
    )
    assert_allclose(actual_dv, expected_dv, rtol=2e-4, atol=2e-5)
    assert_allclose(actual_dx, expected_dx, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize(
    ("v", "x"),
    [(0.0, 0.1), (0.5, 2.0), (2.3, 1.5)],
)
def test_log_bessel_ive_value(v: float, x: float) -> None:
    expected = math.log(sc.ive(v, x))
    assert_allclose(log_bessel_ive(jnp.asarray(v), jnp.asarray(x)), expected, rtol=1e-6)


@pytest.mark.parametrize(
    ("v", "x"),
    [(0.5, 0.8), (2.3, 1.5)],
)
def test_log_bessel_ive_derivatives(v: float, x: float) -> None:
    step = 1e-4
    expected_dv = (math.log(sc.ive(v + step, x)) - math.log(sc.ive(v - step, x))) / (2.0 * step)
    expected_dx = (math.log(sc.ive(v, x + step)) - math.log(sc.ive(v, x - step))) / (2.0 * step)
    actual_dv, actual_dx = jax.jacfwd(log_bessel_ive, argnums=(0, 1))(
        jnp.asarray(v), jnp.asarray(x)
    )
    assert_allclose(actual_dv, expected_dv, rtol=5e-3, atol=5e-4)
    assert_allclose(actual_dx, expected_dx, rtol=2e-5, atol=2e-6)
