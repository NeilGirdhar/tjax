
from jax import vjp
from numpy.testing import assert_equal

from tjax import copy_cotangent, replace_cotangent


def test_copy_cotangent() -> None:
    primals, vjp_f = vjp(copy_cotangent, 1.0, 2.0)
    assert_equal(primals, 1.0)
    assert_equal(vjp_f(3.0), (3.0, 3.0))


def test_replace_cotangent() -> None:
    primals, vjp_f = vjp(replace_cotangent, 1.0, 2.0)
    assert_equal(primals, 1.0)
    assert_equal(vjp_f(3.0), (2.0, 3.0))
