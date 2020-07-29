from typing import Tuple, Type, TypeVar

from jax import numpy as jnp

from .annotations import Tensor
from .dataclass import dataclass

__all__ = ['LogCotangent']

T = TypeVar('T', bound='LogCotangent')


@dataclass
class LogCotangent:
    """
    LogCotangent logs cotangents in a differentiated jax function.  For example:

        from jax import grad
        from jax import numpy as jnp

        from tjax import LogCotangent

        def loss(x, w, log_cotangent):
            y = x * w
            z = log_cotangent.forward(y)
            return jnp.sum(jnp.square(2.0 - z))

        x = jnp.array([1.0, 2.0])
        w = jnp.array([2.2, 3.5])
        lg_bar = grad(loss, 2)(x, w, LogCotangent.create(shape=x.shape))

        # lg_bar.cotangent now holds the transmitted cotangent.
    """
    cotangent: Tensor

    @classmethod
    def create(cls: Type[T], shape: Tuple[int, ...]) -> T:
        """
        Factory to create LogCotangent object.__class__(
        Args:
            shape: The shape of the transmitted tensor and its cotangent.
        """
        return cls(jnp.zeros(shape))

    def forward(self, x: Tensor) -> Tensor:
        """
        This method is called in the forward pass.  It will automatically log the cotangent in the
        backward pass.

        Args:
            x: The tensor to be transmitted.
        Returns: The same tensor that was inputted, x.
        """
        if x.shape != self.cotangent.shape:
            raise ValueError
        return x + self.cotangent
