import numpy as np
from chex import Numeric
from numpy.testing import assert_allclose

from tjax import leaky_integrate


def test_time_step_invariance() -> None:
    def f(n: int) -> Numeric:
        x: Numeric = 0.0
        for i in range(n):
            x = leaky_integrate(x, time_step=1 / n, drift=3.0, decay=2.0)
        return x

    a = np.array([f(n) for n in [1, 5, 20, 100]])
    assert_allclose(a, a[0])  # type: ignore
