from __future__ import annotations

import numpy as np
from jax.random import KeyArray, split

from .annotations import ShapeLike

__all__ = ['vmap_split']


def vmap_split(rng: KeyArray, shape: ShapeLike) -> KeyArray:
    """
    Split a scalar key array into a key array that can be passed to a vmapped function.
    """
    if rng.shape != ():
        raise ValueError("Cannot vmap-split a non-scalar key array.")
    if isinstance(shape, int):
        shape = (shape,)
    else:
        shape = tuple(shape)
    prod_shape = int(np.prod(shape))
    rngs = rng if prod_shape == 1 else split(rng, prod_shape)
    return rngs.reshape(shape)
