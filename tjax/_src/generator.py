from __future__ import annotations

from typing import Any, List, Optional, Type, TypeVar, Union

import jax.numpy as jnp
import jax.random
import numpy as np

from .annotations import Array, BooleanArray, RealArray, RealNumeric, Shape, ShapeLike
from .dataclasses import dataclass

__all__ = ['Generator']


T = TypeVar('T', bound='Generator')
DTypeLikeFloat = Any


@dataclass
class Generator:
    """
    This class represents a JAX random number generator.  Unlike `numpy.Generator`, `tjax.Generator`
    has no mutating methods.  Instead, its generation methods return a new instance along with the
    generated tensor.
    """
    key: jnp.ndarray

    # Class methods --------------------------------------------------------------------------------
    @classmethod
    def from_seed(cls: Type[T], seed: int) -> T:
        return cls(jax.random.PRNGKey(seed))

    # New methods ----------------------------------------------------------------------------------
    def is_vmapped(self) -> bool:
        return self.key.shape != (2,)

    def split(self, n: int = 2) -> List[Generator]:
        """
        Split a generator into a list of generators.
        """
        assert not self.is_vmapped()
        keys = jax.random.split(self.key, n)
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
                else jax.random.split(self.key, prod_shape))
        # https://github.com/google/jax/issues/6473
        keys = keys.reshape(shape + (2,))  # type: ignore
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
        return hash(tuple(np.ravel(self.key)))
