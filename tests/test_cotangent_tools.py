from __future__ import annotations

from functools import partial
from typing import Any

import jax.numpy as jnp
from jax import vjp
from numpy.testing import assert_equal

from tjax import assert_tree_allclose, copy_cotangent, cotangent_combinator, replace_cotangent


def test_copy_cotangent() -> None:
    primals, vjp_f = vjp(copy_cotangent, 1.0, 2.0)
    assert_equal(primals, 1.0)
    assert_equal(vjp_f(3.0), (3.0, 3.0))


def test_replace_cotangent() -> None:
    primals, vjp_f = vjp(replace_cotangent, 1.0, 2.0)
    assert_equal(primals, 1.0)
    assert_equal(vjp_f(3.0), (2.0, 3.0))


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
