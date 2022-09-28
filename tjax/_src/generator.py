from __future__ import annotations

from typing import Any, List, Optional, Type, TypeVar, Union

import jax.numpy as jnp
import jax.random
import numpy as np
from jax.random import KeyArray, PRNGKey, split

from .annotations import Array, BooleanArray, RealArray, RealNumeric, Shape, ShapeLike
from .dataclasses import dataclass

__all__ = ['vmap_split', 'Generator']


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


T = TypeVar('T', bound='Generator')
DTypeLikeFloat = Any


@dataclass
class Generator:
    """
    This class represents a JAX random number generator.  Unlike `numpy.Generator`, `tjax.Generator`
    has no mutating methods.  Instead, its generation methods return a new instance along with the
    generated tensor.
    """
    key: KeyArray

    def __post_init__(self) -> None:
        from warnings import warn
        warn("This is deprecated in favor of jax.PRNGKey, and KeyArray.")

    # Class methods --------------------------------------------------------------------------------
    @classmethod
    def from_seed(cls: Type[T], seed: int) -> T:
        return cls(PRNGKey(seed))

    # New methods ----------------------------------------------------------------------------------
    def is_vmapped(self) -> bool:
        return self.key.shape != (2,)

    def split(self, n: int = 2) -> List[Generator]:
        """
        Split a generator into a list of generators.
        """
        assert not self.is_vmapped()
        keys = split(self.key, n)
        return [Generator(key=key) for key in keys]

    def vmap_split(self, shape: ShapeLike) -> Generator:
        """
        Split a generator into a generator that can be passed to a vmapped function.
        """
        assert not self.is_vmapped()
        if isinstance(shape, int):
            shape = (shape,)
        else:
            shape = tuple(shape)
        prod_shape = int(np.prod(shape))
        keys = (self.key
                if prod_shape == 1
                else split(self.key, prod_shape))
        keys = keys.reshape(shape + (2,))
        return Generator(keys)

    def bernoulli(self, p: RealNumeric, shape: Optional[Shape] = None) -> BooleanArray:
        return jax.random.bernoulli(self.key, p, shape)

    def choice(self,
               a: Union[int, Array],
               shape: Shape = (),
               replace: bool = True,
               p: Optional[RealArray] = None) -> Union[int, Array]:
        return jax.random.choice(self.key, a, shape, replace, p)

    def gamma(self,
              gamma_shape: RealNumeric,
              shape: Optional[Shape] = None,
              dtype: DTypeLikeFloat = np.float64) -> RealArray:
        return jax.random.gamma(self.key, gamma_shape, shape)

    def normal(self,
               shape: Shape = (),
               dtype: DTypeLikeFloat = np.float64) -> jnp.ndarray:
        return jax.random.normal(self.key, shape, dtype)

    def uniform(self,
                shape: Shape = (),
                dtype: DTypeLikeFloat = np.float64,
                minval: RealNumeric = 0.,
                maxval: RealNumeric = 1.) -> RealArray:
        return jax.random.uniform(self.key, shape, dtype, minval, maxval)
