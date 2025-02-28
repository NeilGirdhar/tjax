import math
from functools import partial
from typing import Any

import jax.numpy as jnp
import pytest
from jax import vjp
from numpy.testing import assert_allclose

from tjax import Array, JaxRealArray, divide_where, normalize


@pytest.mark.parametrize(("x", "axis", "result"),
                         [((1.0, 2.0, 3.0, 4.0), 0, (0.1, 0.2, 0.3, 0.4)),
                          ((0.0, 0.0), 0, (0.5, 0.5)),
                          ])
def test_l1(x: Any,
            axis: tuple[int, ...] | int | None,
            result: JaxRealArray,
            ) -> None:
    assert_allclose(normalize('l1', jnp.asarray(x), axis=axis), jnp.asarray(result))


@pytest.mark.parametrize(("x", "axis", "result"),
                         [((5.0, 12.0), 0, (5 / 13, 12 / 13)),
                          ((0.0, 0.0), 0, jnp.ones(2) / jnp.sqrt(2)),
                          ])
def test_l2(x: Any,
            axis: tuple[int, ...] | int | None,
            result: JaxRealArray,
            ) -> None:
    assert_allclose(normalize('l2', jnp.asarray(x), axis=axis), jnp.asarray(result))


@pytest.mark.parametrize("k", [0.0, 1.0])
def test_divide_where(k: float) -> None:
    s = (5, 3)
    w = jnp.ones(s) * k
    x = jnp.arange(math.prod(w.shape), dtype='f').reshape(w.shape)
    dummy = jnp.ones_like(w[..., 0])

    def f(w: Array, x: Array, dummy: Array) -> Array:
        total_w = jnp.sum(w, axis=-1)
        return divide_where(dividend=jnp.sum(w * x, axis=-1),
                            divisor=total_w,
                            where=total_w > 0.0,
                            otherwise=dummy)

    y, vjp_f = vjp(partial(f, dummy=dummy), w, x)
    w_bar, x_bar = vjp_f(jnp.ones_like(y))
    assert_allclose(w_bar, k / s[-1] * jnp.tile(jnp.asarray([-1, 0, 1]), (*s[:-1], 1)), atol=1e-3)
    assert_allclose(x_bar, k / s[-1] * jnp.ones(s))
