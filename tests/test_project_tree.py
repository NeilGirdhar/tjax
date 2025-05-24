from math import prod

import jax.numpy as jnp
import pytest

from tjax import Projector


@pytest.mark.parametrize("shape", [(3, 5), (5, 3), (7, 2), (9, 4, 1), (3, 11, 0)])
def test_projection(shape: tuple[int, ...]) -> None:
    x = jnp.reshape(jnp.arange(prod(shape), dtype=float), shape)
    projector = Projector(seed=0)
    projected = projector.project(x)
    assert projected.shape[:-1] == shape[:-1]
    assert projected.shape[-1] == min(shape[-1], 2)
