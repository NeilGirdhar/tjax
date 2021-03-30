from __future__ import annotations

from typing import Any, List, Optional, Tuple, Type, TypeVar, Union

import jax.numpy as jnp
import jax.random
import numpy as np
from chex import Array

from .annotations import RealArray, Shape, ShapeLike
from .dataclass import dataclass

__all__ = ['Generator']


T = TypeVar('T', bound='Generator')


@dataclass
class Generator:
    """
    This class represents a JAX random number generator.  Unlike `numpy.Generator`, `tjax.Generator`
    has no mutating methods.  Instead, its generation methods return a new instance along with the
    generated tensor.
    """

    key: Array

    # Class methods --------------------------------------------------------------------------------
    @classmethod
    def from_seed(cls: Type[T], seed: int) -> T:
        return cls(jax.random.PRNGKey(seed))

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
        prod_shape = int(np.prod(shape))
        keys = (self.key
                if prod_shape == 1
                else jax.random.split(self.key, prod_shape))
        return Generator(key=keys.reshape(shape + (2,)))

    def bernoulli(self, p: RealArray, shape: Shape = ()) -> Tuple[Array, Generator]:
        new_g, this_g = self.split()
        return jax.random.bernoulli(this_g.key, p, shape), new_g

    def choice(self,
               a: Union[int, Array],
               shape: Shape = (),
               replace: bool = True,
               p: Optional[Array] = None) -> Tuple[Array, Generator]:
        new_g, this_g = self.split()
        return jax.random.choice(self.key, a, shape, replace, p), new_g

    def gamma(self, gamma_shape: RealArray, shape: Shape = ()) -> Tuple[Array, Generator]:
        new_g, this_g = self.split()
        return jax.random.gamma(this_g.key, gamma_shape, shape), new_g

    def normal(self, std_dev: RealArray, shape: Shape = ()) -> Tuple[Array, Generator]:
        new_g, this_g = self.split()
        return std_dev * jax.random.normal(this_g.key, shape), new_g

    def uniform(self, minval: RealArray = 0.0, maxval: RealArray = 0.0, shape: Shape = ()) -> (
            Tuple[Array, Generator]):
        new_g, this_g = self.split()
        return jax.random.uniform(this_g.key, minval=minval, maxval=maxval, shape=shape), new_g

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
