from __future__ import annotations

from typing import Any, List, Optional, Tuple

import jax.random
from jax import numpy as jnp

from .annotations import Shape, Tensor
from .dataclass import dataclass

__all__ = ['Generator']


@dataclass
class Generator:
    """
    This class represents a JAX random number generator.  Unlike `numpy.Generator`, `tjax.Generator`
    has no mutating methods.  Instead, its generation methods return a new instance along with
    the generated tensor.
    """

    key: Tensor

    def __init__(self,
                 *,
                 seed: Optional[int] = None,
                 key: Optional[Tensor] = None,
                 **kwargs: Any):
        super().__init__(**kwargs)
        if key is None:
            if seed is None:
                raise ValueError
            key = jax.random.PRNGKey(seed)
        object.__setattr__(self, 'key', key)

    # New methods ----------------------------------------------------------------------------------
    def split(self, n: int = 2) -> List[Generator]:
        keys = jax.random.split(self.key, n)
        return [Generator(key=key) for key in keys]

    def normal(self, std_dev: Tensor, shape: Shape = ()) -> (
            Tuple[Generator, Tensor]):
        g1, g2 = self.split()
        return g1, std_dev * jax.random.normal(g2.key, shape)

    def gamma(self, gamma_shape: Tensor, shape: Shape = ()) -> (
            Tuple[Generator, Tensor]):
        g1, g2 = self.split()
        return g1, jax.random.gamma(g2.key, gamma_shape, shape)

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
