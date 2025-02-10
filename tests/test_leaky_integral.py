import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.testing import assert_allclose

from tjax import JaxRealArray, leaky_integrate


def test_time_step_invariance() -> None:
    def f(n: int) -> JaxRealArray:
        x = jnp.asarray(0.0)
        for _ in range(n):
            x = leaky_integrate(x,
                                time_step=jnp.asarray(1 / n),
                                drift=jnp.asarray(3.0),
                                decay=jnp.asarray(2.0))
        assert isinstance(x, Array)
        return x

    a = np.asarray([f(n) for n in (1, 5, 20, 100)],
                   dtype=np.float64)
    assert_allclose(a, a[0])
