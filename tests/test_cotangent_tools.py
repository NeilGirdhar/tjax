from __future__ import annotations

from functools import partial

import jax.numpy as jnp
import pytest
from jax import vjp
from numpy.testing import assert_equal

from tjax import (
    JaxRealArray,
    assert_tree_allclose,
    copy_cotangent,
    negate_cotangent,
    print_cotangent,
    replace_cotangent,
    scale_cotangent,
)
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
    (x_bar,) = vjp_f(X(4 * o, 5 * o))
    assert_tree_allclose(x_bar, X(12 * o, 15 * o))


def test_tree_scale() -> None:
    o = jnp.ones(())
    x = X(2 * o, 3 * o)
    tree_scale = X(6 * o, -7 * o)
    x_out, vjp_f = vjp(partial(scale_cotangent, tree_scale=tree_scale), x)
    assert_tree_allclose(x, x_out)
    (x_bar,) = vjp_f(X(4 * o, 5 * o))
    assert_tree_allclose(x_bar, X(24 * o, -35 * o))


def test_scalar_tree_scale() -> None:
    o = jnp.ones(())
    x = X(2 * o, 3 * o)
    tree_scale = X(6 * o, -7 * o)
    x_out, vjp_f = vjp(partial(scale_cotangent, scalar_scale=3.0, tree_scale=tree_scale), x)
    assert_tree_allclose(x, x_out)
    (x_bar,) = vjp_f(X(4 * o, 5 * o))
    assert_tree_allclose(x_bar, X(3 * 24 * o, 3 * -35 * o))


def test_negate_cotangent() -> None:
    o = jnp.ones(())
    x = X(2 * o, 3 * o)
    x_out, vjp_f = vjp(negate_cotangent, x)
    assert_tree_allclose(x_out, x)
    (x_bar,) = vjp_f(X(4 * o, 5 * o))
    assert_tree_allclose(x_bar, X(-4 * o, -5 * o))


def test_replace_cotangent() -> None:
    primals, vjp_f = vjp(replace_cotangent, 1.0, 2.0)
    assert_equal(primals, 1.0)
    assert_equal(vjp_f(3.0), (2.0, 3.0))


def test_replace_cotangent_pytree() -> None:
    o = jnp.ones(())
    x = X(2 * o, 3 * o)
    new_cotangent = X(5 * o, 7 * o)
    output_cotangent = X(11 * o, 13 * o)

    primals, vjp_f = vjp(replace_cotangent, x, new_cotangent)
    assert_tree_allclose(primals, x)
    x_bar, new_cotangent_bar = vjp_f(output_cotangent)
    assert_tree_allclose(x_bar, new_cotangent)
    assert_tree_allclose(new_cotangent_bar, output_cotangent)


def test_copy_cotangent() -> None:
    primals, vjp_f = vjp(copy_cotangent, 1.0, 2.0)
    assert_equal(primals, 1.0)
    assert_equal(vjp_f(3.0), (3.0, 3.0))


def test_copy_cotangent_pytree() -> None:
    o = jnp.ones(())
    x = X(2 * o, 3 * o)
    y = X(5 * o, 7 * o)
    output_cotangent = X(11 * o, 13 * o)

    primals, vjp_f = vjp(copy_cotangent, x, y)
    assert_tree_allclose(primals, x)
    x_bar, y_bar = vjp_f(output_cotangent)
    assert_tree_allclose(x_bar, output_cotangent)
    assert_tree_allclose(y_bar, output_cotangent)


@pytest.mark.parametrize("name", [None, "cotangent"])
def test_print_cotangent(capsys: pytest.CaptureFixture[str], name: str | None) -> None:
    primals, vjp_f = vjp(partial(print_cotangent, name=name), 1.0)
    assert_equal(primals, 1.0)
    assert_equal(vjp_f(3.0), (3.0,))
    captured = capsys.readouterr()
    assert "3.0000" in captured.out
    if name is None:
        assert "cotangent" not in captured.out
    else:
        assert name in captured.out
