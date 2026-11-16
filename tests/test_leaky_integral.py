import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from numpy.testing import assert_allclose

from tjax import JaxRealArray, diffused_leaky_integrate, leaky_integrate


def test_time_step_invariance() -> None:
    def f(n: int) -> JaxRealArray:
        x = jnp.asarray(0.0)
        for _ in range(n):
            x = leaky_integrate(
                x, time_step=jnp.asarray(1 / n), drift=jnp.asarray(3.0), decay=jnp.asarray(2.0)
            )
        assert isinstance(x, Array)
        return x

    a = np.asarray([f(n) for n in (1, 5, 20, 100)], dtype=np.float64)
    assert_allclose(a, a[0])


def test_zero_decay_matches_no_decay() -> None:
    value = jnp.asarray(1.5)
    time_step = jnp.asarray(0.7)
    drift = jnp.asarray(3.0)
    decayed = leaky_integrate(value, time_step, drift, jnp.asarray(0.0))
    undecayed = leaky_integrate(value, time_step, drift)
    assert_allclose(decayed, undecayed)


def test_zero_decay_leaky_average() -> None:
    # An average with no decay never admits data, so the drift contributes nothing.
    value = jnp.asarray(1.5)
    result = leaky_integrate(
        value, jnp.asarray(1.0), jnp.asarray(3.0), jnp.asarray(0.0), leaky_average=True
    )
    assert_allclose(result, value)


@pytest.mark.parametrize("decay", [0.0, 1e-8, 9.99e-4, 1.001e-3, 0.5, 50.0])
def test_decay_value_and_gradient(decay: float) -> None:
    value = 1.5
    drift = 3.0

    def f(k: JaxRealArray) -> JaxRealArray:
        result = leaky_integrate(jnp.asarray(value), jnp.asarray(1.0), jnp.asarray(drift), k)
        assert isinstance(result, Array)
        return result

    # Closed forms of the update and its derivative in the decay, using expm1 for accuracy.
    if decay == 0.0:
        expected = value + drift
        expected_gradient = -value - drift / 2
    else:
        weight = -np.expm1(-decay) / decay
        expected = np.exp(-decay) * value + drift * weight
        expected_gradient = -np.exp(-decay) * value + drift * (np.exp(-decay) - weight) / decay
    k = jnp.asarray(decay)
    assert_allclose(f(k), expected, rtol=1e-6)
    assert_allclose(jax.grad(f)(k), expected_gradient, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("decay", [0.0, 1e-5, 0.5 + 2.0j])
def test_complex_decay(decay: complex) -> None:
    value = jnp.asarray(1.0 + 1.0j)
    drift = jnp.asarray(0.5j)
    result = leaky_integrate(
        value, jnp.asarray(1.0), drift, jnp.asarray(decay, dtype=jnp.complex128)
    )
    weight = 1.0 if decay == 0.0 else -np.expm1(-decay) / decay
    assert_allclose(result, np.exp(-decay) * complex(value) + complex(drift) * weight, rtol=1e-6)


def test_diffused_zero_decay_is_finite() -> None:
    result = diffused_leaky_integrate(
        jnp.zeros(3),
        jnp.asarray(1.0),
        jax.random.key(0),
        jnp.asarray(2.0),
        jnp.asarray(1.0),
        jnp.asarray(0.0),
    )
    assert jnp.all(jnp.isfinite(result))
