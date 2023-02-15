from __future__ import annotations

import math

from jax.random import KeyArray, split

from .annotations import ShapeLike

__all__ = ['vmap_split']


def vmap_split(rng: KeyArray, shape: ShapeLike) -> KeyArray:
    """Split a scalar key array into a key array that can be passed to a vmapped function."""
    if rng.shape != ():
        msg = "Cannot vmap-split a non-scalar key array."
        raise ValueError(msg)
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    prod_shape = math.prod(shape)
    rngs = rng if prod_shape == 1 else split(rng, prod_shape)
    return rngs.reshape(shape)
