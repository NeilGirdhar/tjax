import numpy as np
import pytest
from chex import Numeric
from jax import numpy as jnp
from numpy.testing import assert_allclose

from tjax import leaky_integrate, leaky_integrate_time_series


def test_time_step_invariance() -> None:
    def f(n: int) -> Numeric:
        x: Numeric = 0.0
        for i in range(n):
            x = leaky_integrate(x, time_step=1 / n, drift=3.0, decay=2.0)
        return x

    a = np.array([f(n) for n in [1, 5, 20, 100]])
    assert_allclose(a, a[0])  # type: ignore


@pytest.mark.parametrize('decay', [0.1, 1.0, 10.0])
def test_time_series(decay: float) -> None:
    n = jnp.ones(100) * 12.0
    assert_allclose(n, leaky_integrate_time_series(n, decay))
