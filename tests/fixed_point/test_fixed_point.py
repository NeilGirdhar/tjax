from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Protocol

import jax.numpy as jnp
import pytest
from jax import grad
from jax.random import KeyArray, PRNGKey, normal, split
from numpy.testing import assert_allclose
from typing_extensions import override

from tjax import Array, PyTree
from tjax.dataclasses import dataclass, field
from tjax.fixed_point import (ComparingIteratedFunctionWithCombinator, ComparingState,
                              StochasticIteratedFunctionWithCombinator)


class C(Protocol):
    def __call__(self, theta: Array, x_init: Array) -> Array:
        ...


State = tuple[PyTree, KeyArray]


@dataclass
class NewtonsMethod(ComparingIteratedFunctionWithCombinator[PyTree, Array, Array, Array, None]):
    f: Callable[[PyTree, Array], Array] = field(static=True)
    step_size: float

    # Implemented methods --------------------------------------------------------------------------
    @override
    def sampled_state(self, theta: PyTree, state: Array) -> Array:
        g = grad(self.f, 1)(theta, state)
        ratio = self.f(theta, state) / g
        return state - jnp.where(ratio < 1e6, ratio * self.step_size, 0.0)  # noqa: PLR2004

    @override
    def sampled_state_trajectory(self,
                                 theta: PyTree,
                                 augmented: ComparingState[Array, Array]) -> tuple[Array, None]:
        return self.sampled_state(theta, augmented.current_state), None

    @override
    def extract_comparand(self, state: Array) -> Array:
        return state

    @override
    def extract_differentiand(self, theta: PyTree, state: Array) -> Array:
        return state

    @override
    def implant_differentiand(self, theta: PyTree, state: Array, differentiand: Array) -> Array:
        return differentiand


@dataclass
class NoisyNewtonsMethod(StochasticIteratedFunctionWithCombinator[PyTree, State, Array, Array,
                                                                  Array]):
    f: Callable[[PyTree, Array], Array] = field(static=True)
    step_size: float

    # Implemented methods --------------------------------------------------------------------------
    @override
    def expected_state(self, theta: PyTree, state: State) -> State:
        x, rng = state
        g = grad(self.f, 1)(theta, x)
        ratio = self.f(theta, x) / g
        new_x = x - jnp.where(ratio < 1e6, ratio * self.step_size, 0.0)  # noqa: PLR2004
        return new_x, rng

    @override
    def sampled_state(self, theta: PyTree, state: State) -> State:
        new_x, rng = self.expected_state(theta, state)
        rng, new_rng = split(rng)
        noise = 1e-4 * normal(rng)
        return new_x + noise, new_rng

    @override
    def extract_comparand(self, state: State) -> PyTree:
        x, _ = state
        return x

    @override
    def extract_differentiand(self, theta: PyTree, state: State) -> PyTree:
        x, _ = state
        return x

    @override
    def implant_differentiand(self, theta: PyTree, state: State, differentiand: Array) -> State:
        _, rng = state
        return differentiand, rng


def squared_error(theta: Array, x: Array) -> Array:
    return jnp.square(x - theta)


@pytest.fixture(scope='session', name='it_fun')
def fixture_it_fun() -> NewtonsMethod:
    step_size = 0.01
    return NewtonsMethod(minimum_iterations=11, maximum_iterations=2000, rtol=1e-4,
                         atol=1e-6, z_minimum_iterations=11, z_maximum_iterations=1000,
                         f=squared_error, step_size=step_size)


@pytest.fixture(scope='session', name='fixed_point_using_while')
def fixture_fixed_point_using_while(it_fun: NewtonsMethod) -> C:
    def f(theta: Array, x_init: Array) -> Array:
        return it_fun.find_fixed_point(theta, x_init).current_state
    return f


@pytest.fixture(scope='session', name='fixed_point_using_scan')
def fixture_fixed_point_using_scan(it_fun: NewtonsMethod) -> C:
    def f(theta: Array, x_init: Array) -> Array:
        return it_fun.sample_trajectory(theta, x_init, 2000, None)[0].current_state
    return f


@pytest.fixture(scope='session', name='noisy_it_fun')
def fixture_noisy_it_fun() -> NoisyNewtonsMethod:
    return NoisyNewtonsMethod(minimum_iterations=11,
                              maximum_iterations=1000,
                              rtol=1e-4,
                              atol=1e-4,
                              z_minimum_iterations=11,
                              z_maximum_iterations=1000,
                              convergence_detection_decay=0.05,
                              f=squared_error,
                              step_size=0.05)


@pytest.mark.parametrize('theta', [3.4, 5.8, -9.2])
@pytest.mark.parametrize('x_init', [3.4, 5.8, -9.2])
def test_forward(fixed_point_using_while: C,
                 fixed_point_using_scan: C,
                 theta: Array,
                 x_init: Array) -> None:
    assert_allclose(fixed_point_using_scan(theta, x_init), theta, rtol=1e-1)
    assert_allclose(fixed_point_using_while(theta, x_init), theta, rtol=1e-1)


@pytest.mark.parametrize('theta', [-10.0, -1.0, 0.0, 1.0, 10.0])
def test_grad(fixed_point_using_while: C,
              fixed_point_using_scan: C,
              theta: Array) -> None:
    g = grad(partial(fixed_point_using_while, x_init=8.0))
    h = grad(partial(fixed_point_using_scan, x_init=8.0))
    assert_allclose(1.0, g(theta), rtol=1e-1)
    assert_allclose(1.0, h(theta), rtol=1e-1)


@pytest.mark.parametrize('theta', [-5.0, -1.0, 0.0, 1.0, 5.0])
def test_noisy_grad(noisy_it_fun: NoisyNewtonsMethod, theta: float) -> None:

    def fixed_point_using_while_of_theta(theta: float) -> float:
        state = (8.0, PRNGKey(123))
        x, _ = noisy_it_fun.find_fixed_point(theta, state).current_state
        return x
    assert_allclose(theta,
                    fixed_point_using_while_of_theta(theta),
                    rtol=1e-2,
                    atol=1e-2)
    assert_allclose(1.0,
                    grad(fixed_point_using_while_of_theta)(theta),
                    rtol=1e-2)
