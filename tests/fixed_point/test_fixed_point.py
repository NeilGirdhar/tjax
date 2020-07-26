from functools import partial
from typing import Callable, Tuple

import pytest
from jax import grad
from jax import numpy as jnp
from numpy.testing import assert_allclose

from tjax import Generator, PyTree, Tensor, dataclass, field
from tjax.fixed_point import (ComparingIteratedFunctionWithCombinator, IteratedFunction,
                              StochasticIteratedFunctionWithCombinator)

C = Callable[[Tensor, Tensor], Tensor]


@dataclass
class NewtonsMethod(ComparingIteratedFunctionWithCombinator[PyTree, Tensor, Tensor]):

    f: Callable[[PyTree, Tensor], Tensor] = field(False)
    step_size: float

    # Implemented methods --------------------------------------------------------------------------
    def iterate_state(self, theta: PyTree, x: PyTree) -> PyTree:
        # https://github.com/python/mypy/issues/5485
        g = grad(self.f, 1)(theta, x)  # type: ignore
        ratio = self.f(theta, x) / g  # type: ignore
        return x - jnp.where(ratio < 1e6,
                             ratio * self.step_size,
                             0.0)

    def extract_comparand(self, state: PyTree) -> PyTree:
        return state


@dataclass
class NoisyNewtonsMethod(StochasticIteratedFunctionWithCombinator[PyTree, Tensor, Tensor]):

    f: Callable[[PyTree, Tensor], Tensor] = field(False)
    step_size: float

    # Implemented methods --------------------------------------------------------------------------
    def extract_comparand(self, state: PyTree) -> PyTree:
        return state

    def iterate_state(self, theta: PyTree, x: PyTree) -> PyTree:
        # https://github.com/python/mypy/issues/5485
        g = grad(self.f, 1)(theta, x)  # type: ignore
        ratio = self.f(theta, x) / g  # type: ignore
        return x - jnp.where(ratio < 1e6,
                             ratio * self.step_size,
                             0.0)

    def stochastic_iterate_state(self,
                                 theta: PyTree,
                                 state: Tensor,
                                 rng: Generator) -> Tuple[Tensor, Generator]:
        next_state = self.iterate_state(theta, state)
        new_rng, noise = rng.normal(1e-4)
        return next_state + noise, new_rng


def squared_error(theta: Tensor, x: Tensor) -> Tensor:
    return jnp.square(x - theta)


@pytest.fixture(scope='session', name='it_fun')
def fixture_it_fun() -> NewtonsMethod:
    step_size = 0.01
    return NewtonsMethod(squared_error, step_size, iteration_limit=2000)


@pytest.fixture(scope='session', name='fixed_point_using_while')
def fixture_fixed_point_using_while(it_fun: IteratedFunction[Tensor, Tensor, Tensor]) -> C:
    def f(theta: Tensor, x_init: Tensor) -> Tensor:
        return it_fun.find_fixed_point(theta, x_init).current_state
    return f


@pytest.fixture(scope='session', name='fixed_point_using_scan')
def fixture_fixed_point_using_scan(it_fun: IteratedFunction[Tensor, Tensor, Tensor]) -> C:
    def f(theta: Tensor, x_init: Tensor) -> Tensor:
        return it_fun.sample_trajectory(theta, x_init, 2000, None, lambda x: x)[0].current_state
    return f


@pytest.fixture(scope='session', name='noisy_it_fun')
def fixture_noisy_it_fun() -> NoisyNewtonsMethod:
    return NoisyNewtonsMethod(iteration_limit=1000,
                              atol=1e-4,
                              initial_rng=Generator(seed=123),
                              decay=0.05,
                              f=squared_error,
                              step_size=0.05)


@pytest.mark.parametrize('theta', (3.4, 5.8, -9.2))
@pytest.mark.parametrize('x_init', (3.4, 5.8, -9.2))
def test_forward(fixed_point_using_while: C,
                 fixed_point_using_scan: C,
                 theta: Tensor,
                 x_init: Tensor) -> None:
    assert_allclose(fixed_point_using_scan(theta, x_init),
                    theta,
                    rtol=1e-1)
    assert_allclose(fixed_point_using_while(theta, x_init),
                    theta,
                    rtol=1e-1)


@pytest.mark.parametrize('theta', (-10.0, -1.0, 0.0, 1.0, 10.0))
def test_grad(fixed_point_using_while: C,
              fixed_point_using_scan: C,
              theta: Tensor) -> None:
    g = grad(partial(fixed_point_using_while, x_init=8.0))
    h = grad(partial(fixed_point_using_scan, x_init=8.0))
    assert_allclose(1.0, g(theta), rtol=1e-1)
    assert_allclose(1.0, h(theta), rtol=1e-1)


@pytest.mark.parametrize('theta', (-5.0, -1.0, 0.0, 1.0, 5.0))
def test_noisy_grad(noisy_it_fun: IteratedFunction[Tensor, Tensor, Tensor],
                    theta: float) -> None:

    def fixed_point_using_while_of_theta(theta: Tensor) -> Tensor:
        return noisy_it_fun.find_fixed_point(theta, 8.0).current_state
    assert_allclose(theta,
                    fixed_point_using_while_of_theta(theta),
                    rtol=1e-2,
                    atol=1e-2)
    assert_allclose(1.0,
                    grad(fixed_point_using_while_of_theta)(theta),
                    rtol=1e-2)
