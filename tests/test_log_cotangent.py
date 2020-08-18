from chex import Numeric
from jax import grad
from jax import numpy as jnp

from tjax import LogCotangent, assert_jax_allclose


def test_log_cotangent() -> None:
    def loss(x: Numeric, w: Numeric, log_cotangent: LogCotangent) -> Numeric:
        y = x * w
        z = log_cotangent.forward(y)
        return jnp.sum(jnp.square(2.0 - z))

    x = jnp.array([1.0, 2.0])
    w = jnp.array([2.2, 3.5])
    lg_bar = grad(loss, 2)(x, w, LogCotangent.create(shape=x.shape))
    assert_jax_allclose(lg_bar.cotangent, jnp.array([0.4, 10.0]))
