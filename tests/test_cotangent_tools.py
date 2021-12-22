
from jax import vjp
from numpy.testing import assert_equal

from tjax import copy_cotangent, replace_cotangent


def test_copy_cotangent() -> None:
    p, f = vjp(copy_cotangent, 1.0, 2.0)
    assert_equal(p, (1.0, 2.0))  # type: ignore[no-untyped-call]
    assert_equal(f((3.0, 4.0)), (3.0, 3.0))  # type: ignore[no-untyped-call]


def test_replace_cotangent() -> None:
    p, f = vjp(replace_cotangent, 1.0, 2.0)
    assert_equal(p, 1.0)  # type: ignore[no-untyped-call]
    assert_equal(f(3.0), (2.0, 3.0))  # type: ignore[no-untyped-call]
