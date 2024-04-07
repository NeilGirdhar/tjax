import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from numpy.testing import assert_allclose

from tjax import JaxRealArray, leaky_integrate, leaky_integrate_time_series


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


@pytest.mark.parametrize('decay', [0.1, 1.0, 10.0])
def test_time_series_of_ones(decay: float) -> None:
    n = jnp.ones(100) * 12.0
    assert_allclose(n, leaky_integrate_time_series(n, jnp.asarray(decay)))


def test_time_series() -> None:
    time_series = jnp.asarray([0.1, 0.6, -0.3, 1.5])
    decay = jnp.asarray(0.3)
    result = jnp.asarray([0.1, 0.387221, 0.087076, 0.611119])
    assert_allclose(result, leaky_integrate_time_series(time_series, decay), rtol=3e-6)
