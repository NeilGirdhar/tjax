import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.special import gammainc
from scipy.stats import poisson

from tjax import JaxRealArray, RealNumeric
from tjax._src.filtering.leaky import leaky_integrate  # ruff: ignore[import-private-name]
from tjax.filtering import (
    DelayLine,
    ErlangChain,
    ExponentialBank,
    LeakyAverage,
    LeakyIntegral,
    LeakyMoments,
    LinearFilter,
    StagedFilter,
)


def _erlang_system(decay: RealNumeric, order: int) -> LinearFilter:
    """Return the general linear filter that equals an Erlang chain with unit weights."""
    system = -decay * jnp.eye(order) + jnp.eye(order, k=-1)
    return LinearFilter(jnp.asarray(system), jnp.eye(order)[0], jnp.ones(order))


# Factories, so that the arrays are created under the 64-bit test fixture.
_FILTERS: dict[str, Callable[[], StagedFilter]] = {
    "erlang": lambda: ErlangChain(jnp.asarray(0.7), jnp.asarray([0.0, 1.0, 0.5])),
    "bank": lambda: ExponentialBank(jnp.asarray([0.1, 1.0, 3.0]), jnp.asarray([1.0, 0.5, 0.25])),
    "linear": lambda: _erlang_system(0.7, 3),
}


def _run(staged: StagedFilter, signal: list[float], time_step: float) -> JaxRealArray:
    state = staged.init(jnp.zeros(()))
    for drift in signal:
        state = staged.update(state, jnp.asarray(drift), time_step)
    return staged.value(state)


@pytest.mark.parametrize("name", _FILTERS)
def test_time_step_invariance(name: str) -> None:
    # A signal held over a step equals the same signal over substeps.
    staged = _FILTERS[name]()
    signal = [1.0, -2.0, 0.5]
    coarse = _run(staged, signal, 1.0)
    fine = _run(staged, [drift for drift in signal for _ in range(4)], 0.25)
    assert_allclose(fine, coarse, rtol=1e-12)


def test_first_order_erlang_is_the_leaky_integral() -> None:
    decay = jnp.asarray(0.3)
    chain = ErlangChain(decay, jnp.ones(1))
    stages = chain.update(chain.init(jnp.asarray(2.0)), jnp.asarray(2.0))
    stages = chain.update(stages, jnp.asarray(-1.0))
    expected = leaky_integrate(jnp.asarray(0.0), 1.0, jnp.asarray(2.0), decay)
    expected = leaky_integrate(expected, 1.0, jnp.asarray(-1.0), decay)
    assert_allclose(chain.value(stages), expected)
    assert_allclose(LeakyIntegral(decay).lag_weights(5), chain.lag_weights(5), rtol=1e-6)


@pytest.mark.parametrize("decay", [0.0, 1e-8, 1e-3, 0.5, 6.999999, 7.000001, 30.0])
def test_erlang_matches_matrix_exponential(decay: float) -> None:
    order = 3
    transition, input_weight = ErlangChain(jnp.asarray(decay), jnp.ones(order)).discretize(0.8)
    expected_transition, expected_input = _erlang_system(decay, order).discretize(0.8)
    assert_allclose(transition, expected_transition, rtol=1e-9, atol=1e-15)
    assert_allclose(input_weight, expected_input, rtol=1e-9, atol=1e-15)


@pytest.mark.parametrize("decay", [0.0, 1e-8, 0.5, 6.999999, 7.000001, 30.0])
def test_erlang_gradient_is_finite_and_matches(decay: float) -> None:
    order = 3

    def total(k: JaxRealArray) -> JaxRealArray:
        return jnp.sum(ErlangChain(k, jnp.ones(order)).discretize(1.0)[1])

    def expected_total(k: JaxRealArray) -> JaxRealArray:
        return jnp.sum(_erlang_system(k, order).discretize(1.0)[1])

    k = jnp.asarray(decay)
    assert_allclose(jax.grad(total)(k), jax.grad(expected_total)(k), rtol=1e-6, atol=1e-12)


def test_complex_erlang_matches_matrix_exponential() -> None:
    decay = jnp.asarray(0.5 + 2.0j)
    order = 3
    _, input_weight = ErlangChain(decay, jnp.ones(order, dtype=decay.dtype)).discretize(1.0)
    system = -decay * jnp.eye(order) + jnp.eye(order, k=-1)
    input_gain = jnp.eye(order, dtype=decay.dtype)[0]
    _, expected = LinearFilter(system, input_gain, input_gain).discretize(1.0)
    assert_allclose(input_weight, expected, rtol=1e-9)


@pytest.mark.parametrize("stage", [1, 2, 3])
def test_erlang_lag_weights_integrate_the_kernel(stage: int) -> None:
    decay = 0.7
    num_lags = 6
    basis = ErlangChain(jnp.asarray(decay), jnp.ones(3)).basis_lag_weights(num_lags)
    # The integral of u^(j-1) exp(-k u) / (j-1)! over [L, L + 1] in terms of the gamma CDF.
    lags = np.arange(num_lags)
    expected = (gammainc(stage, decay * (lags + 1)) - gammainc(stage, decay * lags)) / decay**stage
    assert_allclose(basis[stage - 1], expected, rtol=1e-9)


def test_soft_window_is_a_poisson_tail() -> None:
    decay = 1.5
    order = 4
    weights = ErlangChain.soft_window(decay, order).lag_weights(6)
    expected = [
        quad(lambda u: poisson.cdf(order - 1, decay * u), lag, lag + 1)[0] for lag in range(6)
    ]
    assert_allclose(weights, expected, rtol=1e-7)


def test_laguerre_components_are_orthonormal() -> None:
    decay = 0.8
    order = 4
    matrix = np.asarray(ErlangChain(jnp.asarray(decay), jnp.ones(order)).laguerre_matrix())

    def kernel(m: int, u: float) -> float:
        stages = [u**i * math.exp(-decay * u) / math.factorial(i) for i in range(order)]
        return float(matrix[m] @ np.asarray(stages))

    gram = [
        [quad(lambda u, m=m, n=n: kernel(m, u) * kernel(n, u), 0, np.inf)[0] for n in range(order)]
        for m in range(order)
    ]
    assert_allclose(gram, np.eye(order), atol=1e-8)


def test_exponential_bank_matches_matrix_exponential() -> None:
    decays = jnp.asarray([0.0, 0.2, 2.0])
    bank = ExponentialBank(decays, jnp.ones(3))
    general = LinearFilter(-jnp.diag(decays), jnp.ones(3), jnp.ones(3))
    stages = jnp.asarray([[1.0], [2.0], [3.0]])
    drift = jnp.asarray([0.5])
    assert_allclose(bank.update(stages, drift, 0.7), general.update(stages, drift, 0.7), rtol=1e-9)


def test_singular_system_steps_exactly() -> None:
    system = LinearFilter(jnp.zeros((2, 2)), jnp.asarray([1.0, 2.0]), jnp.ones(2))
    transition, input_weight = system.discretize(0.5)
    assert_allclose(transition, np.eye(2))
    assert_allclose(input_weight, [0.5, 1.0])


def test_pytree_signals() -> None:
    chain = ErlangChain.soft_window(1.0, 2)
    template = {"a": jnp.zeros(3), "b": jnp.zeros((2, 2))}
    state = chain.init(template)
    assert state["a"].shape == (2, 3)
    assert state["b"].shape == (2, 2, 2)
    state = chain.update(state, {"a": jnp.ones(3), "b": jnp.ones((2, 2))})
    value = chain.value(state)
    assert value["a"].shape == (3,)
    assert value["b"].shape == (2, 2)
    assert_allclose(value["b"], np.full((2, 2), chain.lag_weights(1)[0]))


def test_filters_pass_through_jit() -> None:
    @jax.jit
    def step(staged: ErlangChain, state: JaxRealArray, drift: JaxRealArray) -> JaxRealArray:
        return staged.update(state, drift)

    chain = ErlangChain.soft_window(1.0, 3)
    state = chain.init(jnp.zeros(2))
    assert_allclose(step(chain, state, jnp.ones(2)), chain.update(state, jnp.ones(2)))


def test_leaky_average_is_unbiased_from_the_first_step() -> None:
    average = LeakyAverage(jnp.asarray(0.1))
    state = average.init(jnp.zeros(2))
    for _ in range(3):
        state = average.update(state, jnp.asarray([2.0, -1.0]))
        assert_allclose(average.value(state), [2.0, -1.0], rtol=1e-6)


def test_leaky_moments() -> None:
    moments = LeakyMoments(jnp.asarray(0.5))
    signal = jnp.asarray([1.0, 2.0])
    state = moments.init(signal)
    state = moments.update(state, signal)
    mean, second = moments.value(state)
    assert_allclose(mean, signal, rtol=1e-6)
    assert_allclose(second, np.outer(signal, signal), rtol=1e-6)


def test_delay_line_holds_recent_lags() -> None:
    line = DelayLine(4)
    state = line.init(jnp.zeros(()))
    for drift in [1.0, 2.0, 3.0]:
        state = line.update(state, jnp.asarray(drift))
    assert_allclose(line.value(state), [3.0, 2.0, 1.0, 0.0])


def test_laguerre_applies_the_laguerre_matrix() -> None:
    chain = ErlangChain(jnp.asarray(0.8), jnp.ones(3))
    stages = jnp.arange(6.0).reshape(3, 2)
    expected = np.asarray(chain.laguerre_matrix()) @ np.asarray(stages)
    assert_allclose(chain.laguerre(stages), expected, rtol=1e-12)
