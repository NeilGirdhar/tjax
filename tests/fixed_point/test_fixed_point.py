from functools import partial
from typing import Callable, Tuple

import pytest
from chex import Array
from jax import grad
from jax import numpy as jnp
from numpy.testing import assert_allclose

from tjax import Generator, PyTree, dataclass, field
from tjax.fixed_point import (ComparingIteratedFunctionWithCombinator, ComparingState,
                              IteratedFunction, StochasticIteratedFunctionWithCombinator)

C = Callable[[Array, Array], Array]


State = Tuple[PyTree, Generator]


@dataclass
class NewtonsMethod(ComparingIteratedFunctionWithCombinator[PyTree, Array, Array, Array, Array]):

    f: Callable[[PyTree, Array], Array] = field(static=True)
    step_size: float

    # Implemented methods --------------------------------------------------------------------------
    def sampled_state(self, theta: PyTree, state: Array) -> Array:
        # https://github.com/python/mypy/issues/5485
        g = grad(self.f, 1)(theta, state)  # type: ignore
        ratio = self.f(theta, state) / g  # type: ignore
        return state - jnp.where(ratio < 1e6, ratio * self.step_size, 0.0)

    def sampled_state_trajectory(self,
                                 theta: PyTree,
                                 augmented: ComparingState[Array, Array]) -> Tuple[Array, None]:
        return self.sampled_state(theta, augmented.current_state), None

    def extract_comparand(self, state: Array) -> Array:
        return state

    def extract_differentiand(self, state: Array) -> Array:
        return state

    def implant_differentiand(self, state: Array, differentiand: Array) -> Array:
        return differentiand


@dataclass
class NoisyNewtonsMethod(StochasticIteratedFunctionWithCombinator[PyTree, State, Array, Array,
                                                                  Array]):

    f: Callable[[PyTree, Array], Array] = field(static=True)
    step_size: float

    # Implemented methods --------------------------------------------------------------------------
    def expected_state(self, theta: PyTree, state: State) -> State:
        x, rng = state
        # https://github.com/python/mypy/issues/5485
        g = grad(self.f, 1)(theta, x)  # type: ignore
        ratio = self.f(theta, x) / g  # type: ignore
        new_x = x - jnp.where(ratio < 1e6, ratio * self.step_size, 0.0)
        return new_x, rng

    def sampled_state(self, theta: PyTree, state: State) -> State:
        new_x, rng = self.expected_state(theta, state)
        noise, new_rng = rng.normal(1e-4)
        return new_x + noise, new_rng

    def extract_comparand(self, state: State) -> PyTree:
        x, rng = state
        return x

    def extract_differentiand(self, state: State) -> PyTree:
        x, rng = state
        return x

    def implant_differentiand(self, state: State, differentiand: Array) -> State:
        x, rng = state
        return differentiand, rng


def squared_error(theta: Array, x: Array) -> Array:
    return jnp.square(x - theta)


@pytest.fixture(scope='session', name='it_fun')
def fixture_it_fun() -> NewtonsMethod:
    step_size = 0.01
    return NewtonsMethod(f=squared_error, step_size=step_size, iteration_limit=2000)


@pytest.fixture(scope='session', name='fixed_point_using_while')
def fixture_fixed_point_using_while(
        it_fun: IteratedFunction[Array, Array, Array, Array, Array]) -> C:
    def f(theta: Array, x_init: Array) -> Array:
        return it_fun.find_fixed_point(theta, x_init).current_state
    return f


@pytest.fixture(scope='session', name='fixed_point_using_scan')
def fixture_fixed_point_using_scan(
    it_fun: IteratedFunction[Array, Array, Array, Array, Array]) -> C:
    def f(theta: Array, x_init: Array) -> Array:
        return it_fun.sample_trajectory(theta, x_init, 2000, None)[0].current_state
    return f


@pytest.fixture(scope='session', name='noisy_it_fun')
def fixture_noisy_it_fun() -> NoisyNewtonsMethod:
    return NoisyNewtonsMethod(iteration_limit=1000,
                              atol=1e-4,
                              convergence_detection_decay=0.05,
                              f=squared_error,
                              step_size=0.05)


@pytest.mark.parametrize('theta', (3.4, 5.8, -9.2))
@pytest.mark.parametrize('x_init', (3.4, 5.8, -9.2))
def test_forward(fixed_point_using_while: C,
                 fixed_point_using_scan: C,
                 theta: Array,
                 x_init: Array) -> None:
    assert_allclose(fixed_point_using_scan(theta, x_init),
                    theta,
                    rtol=1e-1)
    assert_allclose(fixed_point_using_while(theta, x_init),
                    theta,
                    rtol=1e-1)


@pytest.mark.parametrize('theta', (-10.0, -1.0, 0.0, 1.0, 10.0))
def test_grad(fixed_point_using_while: C,
              fixed_point_using_scan: C,
              theta: Array) -> None:
    g = grad(partial(fixed_point_using_while, x_init=8.0))
    h = grad(partial(fixed_point_using_scan, x_init=8.0))
    assert_allclose(1.0, g(theta), rtol=1e-1)
    assert_allclose(1.0, h(theta), rtol=1e-1)


@pytest.mark.parametrize('theta', (-5.0, -1.0, 0.0, 1.0, 5.0))
def test_noisy_grad(noisy_it_fun: IteratedFunction[Array, Array, Array, Array, Array],
                    theta: float) -> None:

    def fixed_point_using_while_of_theta(theta: Array) -> Array:
        state = (8.0, Generator.from_seed(123))
        x, rng = noisy_it_fun.find_fixed_point(theta, state).current_state
        return x
    assert_allclose(theta,
                    fixed_point_using_while_of_theta(theta),
                    rtol=1e-2,
                    atol=1e-2)
    assert_allclose(1.0,
                    grad(fixed_point_using_while_of_theta)(theta),
                    rtol=1e-2)
