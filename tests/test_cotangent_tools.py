from __future__ import annotations

from functools import partial
from typing import Any

import jax.numpy as jnp
from jax import vjp
from numpy.testing import assert_equal

from tjax import (JaxRealArray, assert_tree_allclose, copy_cotangent, cotangent_combinator,
                  replace_cotangent, reverse_scale_cotangent, scale_cotangent)
from tjax.dataclasses import dataclass


@dataclass
class X:
    x: JaxRealArray
    y: JaxRealArray


def test_scalar_scale() -> None:
    o = jnp.ones(())
    x = X(2 * o, 3 * o)
    x_out, vjp_f = vjp(partial(scale_cotangent, scalar_scale=3.0), x)
    assert_tree_allclose(x, x_out)
    x_bar, = vjp_f(X(4 * o, 5 * o))
    assert_tree_allclose(x_bar, X(12 * o, 15 * o))


def test_tree_scale() -> None:
    o = jnp.ones(())
    x = X(2 * o, 3 * o)
    tree_scale = X(6 * o, -7 * o)
    x_out, vjp_f = vjp(partial(scale_cotangent, tree_scale=tree_scale), x)
    assert_tree_allclose(x, x_out)
    x_bar, = vjp_f(X(4 * o, 5 * o))
    assert_tree_allclose(x_bar, X(24 * o, -35 * o))


def test_scalar_tree_scale() -> None:
    o = jnp.ones(())
    x = X(2 * o, 3 * o)
    tree_scale = X(6 * o, -7 * o)
    x_out, vjp_f = vjp(partial(scale_cotangent, scalar_scale=3.0, tree_scale=tree_scale), x)
    assert_tree_allclose(x, x_out)
    x_bar, = vjp_f(X(4 * o, 5 * o))
    assert_tree_allclose(x_bar, X(3 * 24 * o, 3 * -35 * o))


def test_reverse_scale() -> None:
    z = jnp.zeros(())
    o = jnp.ones(())
    x = X(2 * o, 3 * o)
    (x_out, zero), vjp_f = vjp(reverse_scale_cotangent, x)
    assert_tree_allclose(x, x_out)
    assert_tree_allclose(z, zero)
    x_bar, = vjp_f((X(4 * o, 5 * o), 2.0))
    assert_tree_allclose(x_bar, X(8 * o, 10 * o))


def test_replace_cotangent() -> None:
    primals, vjp_f = vjp(replace_cotangent, 1.0, 2.0)
    assert_equal(primals, 1.0)
    assert_equal(vjp_f(3.0), (2.0, 3.0))


def test_copy_cotangent() -> None:
    primals, vjp_f = vjp(copy_cotangent, 1.0, 2.0)
    assert_equal(primals, 1.0)
    assert_equal(vjp_f(3.0), (3.0, 3.0))


def test_combinator() -> None:
    def f(x: Any) -> tuple[Any, None]:
        return (x ** 2, x ** 2), None
    o = jnp.ones(())

    args = ((-1.0,), (-1.0,))
    _, f_vjp = vjp(partial(cotangent_combinator, f, aux_cotangent_scales=None), args)
    result_bar = (2 * o, 3 * o), None
    actual_args_bar, = f_vjp(result_bar)
    desired_args_bar = ((-4.0,), (-6.0,))
    assert_tree_allclose(actual_args_bar, desired_args_bar)
