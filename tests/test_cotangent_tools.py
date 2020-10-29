from functools import partial

import numpy as np
from chex import Array
from jax import vjp
from numpy.testing import assert_equal

from tjax import block_cotangent, copy_cotangent, replace_cotangent


def test_copy_cotangent() -> None:
    p, f = vjp(copy_cotangent, 1.0, 2.0)
    assert_equal(p, (1.0, 2.0))
    assert_equal(f((3.0, 4.0)), (3.0, 3.0))


def test_replace_cotangent() -> None:
    p, f = vjp(replace_cotangent, 1.0, 2.0)
    assert_equal(p, 1.0)
    assert_equal(f(3.0), (2.0, 3.0))


def test_block_cotangent() -> None:

    @partial(block_cotangent, block_argnums=(1,))
    def f(x: Array, y: Array) -> Array:
        return x + y

    p, g = vjp(f, 1.0, 2.0)
    assert_equal(p, np.array(3.0))
    assert_equal(g(4.0), (4.0, 0.0))
