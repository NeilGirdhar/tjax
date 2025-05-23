from typing import Any

import jax.numpy as jnp
import pytest

from tjax import element_count, tree_sum


@pytest.mark.parametrize(("x", "s"),
                         [(jnp.arange(3, 10), 42),
                          ((jnp.arange(5, 9), jnp.arange(4, 12)), 86),
                          ((), 0)])
def test_tree_sum(x: Any, s: float) -> None:
    t = tree_sum(x)
    assert t.ndim == 0
    assert float(t) == s


@pytest.mark.parametrize(("x", "s"),
                         [(jnp.arange(3, 10), 7),
                          ((jnp.arange(5, 9), jnp.arange(4, 12)), 12),
                          ((), 0)])
def test_element_count(x: Any, s: int) -> None:
    ec = element_count(x)
    assert ec == s
