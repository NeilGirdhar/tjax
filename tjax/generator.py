from __future__ import annotations

from typing import Any, List, Optional, Tuple

import jax.random
import numpy as np
from chex import Array
from jax import numpy as jnp

from .annotations import RealArray, Shape, ShapeLike
from .dataclass import dataclass

__all__ = ['Generator']


@dataclass
class Generator:
    """
    This class represents a JAX random number generator.  Unlike `numpy.Generator`, `tjax.Generator`
    has no mutating methods.  Instead, its generation methods return a new instance along with the
    generated tensor.
    """

    key: Array

    def __init__(self,
                 *,
                 seed: Optional[int] = None,
                 key: Optional[Array] = None,
                 **kwargs: Any):
        super().__init__(**kwargs)
        if key is None:
            if seed is None:
                raise ValueError
            key = jax.random.PRNGKey(seed)
        object.__setattr__(self, 'key', key)

    # New methods ----------------------------------------------------------------------------------
    def split(self, n: int = 2) -> List[Generator]:
        """
        Split a generator into a list of generators.
        """
        assert self.key.shape == (2,)
        keys = jax.random.split(self.key, n)
        return [Generator(key=key) for key in keys]

    def vmap_split(self, shape: ShapeLike) -> Generator:
        """
        Split a generator into a generator that can be passed to a vmapped function.
        """
        if isinstance(shape, int):
            shape = (shape,)
        else:
            shape = tuple(shape)
        prod_shape = np.prod(shape)
        keys = (self.key
                if prod_shape == 1
                else jax.random.split(self.key, prod_shape))
        return Generator(key=keys.reshape(shape + (2,)))

    def bernoulli(self, p: RealArray, shape: Shape = ()) -> Tuple[Generator, Array]:
        g1, g2 = self.split()
        return g1, jax.random.bernoulli(g2.key, p, shape)

    def gamma(self, gamma_shape: RealArray, shape: Shape = ()) -> Tuple[Generator, Array]:
        g1, g2 = self.split()
        return g1, jax.random.gamma(g2.key, gamma_shape, shape)

    def normal(self, std_dev: RealArray, shape: Shape = ()) -> Tuple[Generator, Array]:
        g1, g2 = self.split()
        return g1, std_dev * jax.random.normal(g2.key, shape)

    def uniform(self, minval: RealArray = 0.0, maxval: RealArray = 0.0, shape: Shape = ()) -> (
            Tuple[Generator, Array]):
        g1, g2 = self.split()
        return g1, jax.random.uniform(g2.key, minval=minval, maxval=maxval, shape=shape)

    # Magic methods --------------------------------------------------------------------------------
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Generator):
            return NotImplemented
        return jnp.all(self.key == other.key)

    def __ne__(self, other: Any) -> bool:
        if not isinstance(other, Generator):
            return NotImplemented
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash((int(self.key[0]), int(self.key[1])))
