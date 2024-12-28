from typing import Any

import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from tjax import JaxRealArray, normalize


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
