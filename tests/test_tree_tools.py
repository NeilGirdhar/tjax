import jax.numpy as jnp
import pytest
from jax import jit

from tjax import assert_tree_allclose, dynamic_tree_all, element_count, scale_tree, tree_sum


@pytest.mark.parametrize(
    ("x", "s"),
    [
        (jnp.arange(3, 10), 42),
        ((jnp.arange(5, 9), jnp.arange(4, 12)), 86),
        ({"x": jnp.arange(6).reshape(2, 3), "y": (jnp.asarray(-1), jnp.asarray(2))}, 16),
        ((), 0),
    ],
)
def test_tree_sum(x: object, s: float) -> None:
    t = tree_sum(x)
    assert t.ndim == 0
    assert float(t) == s


@pytest.mark.parametrize(
    ("x", "s"),
    [
        (jnp.arange(3, 10), 7),
        ((jnp.arange(5, 9), jnp.arange(4, 12)), 12),
        ({"x": jnp.arange(6).reshape(2, 3), "y": (jnp.asarray(-1), jnp.asarray(2))}, 8),
        ((), 0),
    ],
)
def test_element_count(x: object, s: int) -> None:
    ec = element_count(x)
    assert ec == s


@pytest.mark.parametrize(
    ("x", "expected"),
    [
        ((jnp.ones((), dtype=bool), jnp.ones((), dtype=bool)), True),
        ((jnp.ones((), dtype=bool), jnp.zeros((), dtype=bool)), False),
        ((), True),
    ],
)
def test_dynamic_tree_all(x: object, *, expected: bool) -> None:
    assert bool(dynamic_tree_all(x)) is expected


def test_dynamic_tree_all_jit() -> None:
    @jit
    def f(x: tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
        return dynamic_tree_all(x)

    actual = f((jnp.ones((), dtype=bool), jnp.zeros((), dtype=bool)))
    assert bool(actual) is False


def test_scale_tree_scalar() -> None:
    x = (jnp.arange(3), jnp.arange(3, 5))
    expected = (jnp.asarray([0, 2, 4]), jnp.asarray([6, 8]))
    assert_tree_allclose(scale_tree(x, scalar_scale=2), expected)


def test_scale_tree_tree() -> None:
    x = (jnp.arange(3), jnp.arange(3, 5))
    tree_scale = (jnp.asarray([1, -1, 2]), jnp.asarray([3, 4]))
    expected = (jnp.asarray([0, -1, 4]), jnp.asarray([9, 16]))
    assert_tree_allclose(scale_tree(x, tree_scale=tree_scale), expected)


def test_scale_tree_scalar_and_tree() -> None:
    x = (jnp.arange(3), jnp.arange(3, 5))
    tree_scale = (jnp.asarray([1, -1, 2]), jnp.asarray([3, 4]))
    expected = (jnp.asarray([0, -3, 12]), jnp.asarray([27, 48]))
    assert_tree_allclose(scale_tree(x, scalar_scale=3, tree_scale=tree_scale), expected)
