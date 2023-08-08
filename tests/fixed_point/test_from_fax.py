from __future__ import annotations

from typing import Callable, Tuple

import hypothesis.extra.numpy
import hypothesis.strategies
import jax.numpy as jnp
import jax.scipy
import jax.test_util
import numpy as np
import pytest
from hypothesis import given, settings
from jax import grad, vjp
from numpy.random import Generator
from numpy.testing import assert_allclose
from typing_extensions import override

from tjax import RealArray
from tjax.dataclasses import dataclass
from tjax.fixed_point import ComparingIteratedFunction, ComparingIteratedFunctionWithCombinator


def generate_stable_matrix(generator: Generator,
                           size: int,
                           eps: float = 1e-2) -> RealArray:
    """Generate a random matrix who's singular values are less than 1 - `eps`.

    Args:
        generator: The random number generator.
        size: The size of the matrix. The dimensions of the matrix will be `size`x`size`.
        eps: A float between 0 and 1. The singular values will be no larger than 1 - `eps`.

    Returns:
        A `size`x`size` matrix with singular values less than 1 - `eps`.
    """
    mat = generator.random((size, size))
    return make_stable(mat, eps)


def make_stable(matrix: RealArray, eps: float) -> RealArray:
    u, s, vt = np.linalg.svd(matrix)
    s = np.clip(s, 0, 1 - eps)
    return u.dot(s[:, None] * vt)


def solve_ax_b(amat: RealArray, bvec: RealArray) -> RealArray:
    """Solve for the fixed point x = Ax + b.

    Args:
        amat: A contractive matrix.
        bvec: The vector offset.

    Returns:
        A vector `x` such that x = Ax + b.
    """
    matrix = np.eye(amat.shape[0]) - amat
    return np.linalg.solve(matrix, bvec)


def solve_grad_ax_b(amat: RealArray, bvec: RealArray) -> tuple[RealArray, RealArray]:
    """Solve for the gradient of the fixed point x = Ax + b.

    Args:
        amat: A contractive matrix.
        bvec: The vector offset.

    Returns:
        3-D array: The partial derivative of the fixed point for a given element of `amat`.
        2-D array: The partial derivative of the fixed point for a given element of `bvec`.
    """
    matrix = jnp.eye(amat.shape[0]) - amat
    grad_bvec = jnp.linalg.solve(matrix.T, jnp.ones(matrix.shape[0]))
    grad_matrix = grad_bvec[:, None] * jnp.linalg.solve(matrix, bvec)[None, :]
    return grad_matrix, grad_bvec


TPair = Tuple[RealArray, RealArray]


@dataclass
class Solver(ComparingIteratedFunctionWithCombinator[TPair, RealArray, RealArray, RealArray,
                                                     RealArray]):
    @override
    def sampled_state(self, theta: TPair, state: RealArray) -> RealArray:
        matrix, offset = theta
        return jnp.tensordot(matrix, state, 1) + offset

    @override
    def extract_comparand(self, state: RealArray) -> RealArray:
        return state

    @override
    def extract_differentiand(self, theta: TPair, state: RealArray) -> RealArray:
        return state

    @override
    def implant_differentiand(self,
                              theta: TPair,
                              state: RealArray,
                              differentiand: RealArray) -> RealArray:
        return differentiand


@pytest.fixture(scope='session', name='ax_plus_b')
def fixture_ax_plus_b() -> ComparingIteratedFunctionWithCombinator[TPair, RealArray, RealArray,
                                                                   RealArray, RealArray]:
    return Solver(minimum_iterations=11,
                  maximum_iterations=10000,
                  rtol=1e-10,
                  atol=1e-10,
                  z_minimum_iterations=11,
                  z_maximum_iterations=10000)


# Tests --------------------------------------------------------------------------------------------


@settings(max_examples=100, deadline=5000.)
@given(
    hypothesis.extra.numpy.arrays(
        np.float64, (5, 5), elements=hypothesis.strategies.floats(0, 1)),
    hypothesis.extra.numpy.arrays(
        np.float64, 5, elements=hypothesis.strategies.floats(0, 1)))
def test_simple_contraction(ax_plus_b: ComparingIteratedFunction[TPair, RealArray, RealArray,
                                                                 RealArray],
                            matrix: RealArray,
                            offset: RealArray) -> None:
    matrix = make_stable(matrix, eps=1e-1)
    x0 = jnp.zeros_like(offset)

    true_sol = solve_ax_b(matrix, offset)
    sol = ax_plus_b.find_fixed_point((matrix, offset), x0).current_state

    assert_allclose(sol, true_sol, rtol=1e-5, atol=1e-5)


@settings(max_examples=100, deadline=5000.)
@given(
    hypothesis.extra.numpy.arrays(
        np.float64, (5, 5), elements=hypothesis.strategies.floats(0.1, 1)),
    hypothesis.extra.numpy.arrays(
        np.float64, 5, elements=hypothesis.strategies.floats(0.1, 1)))
def test_jvp(ax_plus_b: ComparingIteratedFunction[TPair, RealArray, RealArray, RealArray],
             matrix: RealArray,
             offset: RealArray) -> None:
    matrix = make_stable(matrix, eps=1e-1)
    x0 = jnp.zeros_like(offset)

    def f(theta: TPair, x_init: RealArray) -> RealArray:
        return ax_plus_b.find_fixed_point(theta, x_init).current_state

    def f_vjp(theta: TPair, x_init: RealArray) -> tuple[RealArray, Callable[[RealArray], TPair]]:
        return vjp(f, theta, x_init)

    jax.test_util.check_vjp(f, f_vjp, ((matrix, offset), x0),  # type: ignore[no-untyped-call]
                            rtol=1e-4, atol=1e-4)


def test_gradient(generator: Generator,
                  ax_plus_b: ComparingIteratedFunction[TPair, RealArray, RealArray,
                                                       RealArray]) -> None:
    """Test gradient on the fixed point of Ax + b = x."""
    mat_size = 10
    matrix = generate_stable_matrix(generator, mat_size, 1e-1)
    offset = generator.uniform(size=mat_size)
    x0 = jnp.zeros_like(offset)

    def loss(params: TPair, x: RealArray) -> RealArray:
        return jnp.sum(ax_plus_b.find_fixed_point(params, x).current_state)

    jax.test_util.check_grads(loss,  # type: ignore[no-untyped-call]
                              ((matrix, offset), x0),
                              order=1,
                              modes=["rev"],
                              atol=1e-4,
                              rtol=1e-4)

    grad_matrix, grad_offset = grad(loss)((matrix, offset), x0)

    true_grad_matrix, true_grad_offset = solve_grad_ax_b(matrix, offset)

    assert_allclose(grad_matrix, true_grad_matrix, rtol=1e-4, atol=1e-5)
    assert_allclose(grad_offset, true_grad_offset, rtol=1e-4, atol=1e-5)
