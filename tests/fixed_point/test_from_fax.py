from typing import Callable, Tuple

import hypothesis.extra.numpy
import jax
import jax.scipy
import jax.test_util
import numpy as np
import pytest
from chex import Array
from jax import numpy as jnp
from numpy.random import Generator
from numpy.testing import assert_allclose

from tjax import dataclass
from tjax.fixed_point import ComparingIteratedFunctionWithCombinator, IteratedFunction


def generate_stable_matrix(generator: Generator,
                           size: int,
                           eps: float = 1e-2) -> np.ndarray:
    """
    Generate a random matrix who's singular values are less than 1 - `eps`.

    Args:
        size: The size of the matrix. The dimensions of the matrix will be `size`x`size`.
        eps: A float between 0 and 1. The singular values will be no larger than 1 - `eps`.

    Returns:
        A `size`x`size` matrix with singular values less than 1 - `eps`.
    """
    mat = generator.random((size, size))
    return make_stable(mat, eps)


def make_stable(matrix: np.ndarray, eps: float) -> np.ndarray:
    u, s, vt = np.linalg.svd(matrix)
    s = np.clip(s, 0, 1 - eps)
    return u.dot(s[:, None] * vt)


def solve_ax_b(amat: np.ndarray, bvec: np.ndarray) -> np.ndarray:
    """
    Solve for the fixed point x = Ax + b.

    Args:
        amat: A contractive matrix.
        bvec: The vector offset.

    Returns:
        A vector `x` such that x = Ax + b.
    """
    matrix = np.eye(amat.shape[0]) - amat
    return np.linalg.solve(matrix, bvec)


def solve_grad_ax_b(amat: np.ndarray, bvec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve for the gradient of the fixed point x = Ax + b.

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


TPair = Tuple[Array, Array]


@dataclass
class Solver(ComparingIteratedFunctionWithCombinator[TPair, Array, Array, Array, Array]):

    def sampled_state(self, theta: TPair, x: Array) -> Array:
        matrix, offset = theta
        return jnp.tensordot(matrix, x, 1) + offset

    def extract_comparand(self, state: Array) -> Array:
        return state

    def extract_differentiand(self, state: Array) -> Array:
        return state

    def implant_differentiand(self, state: Array, differentiand: Array) -> Array:
        return differentiand


@pytest.fixture(name='ax_plus_b', scope='session')
def fixture_ax_plus_b() -> ComparingIteratedFunctionWithCombinator[TPair, Array, Array, Array,
                                                                   Array]:
    return Solver(minimum_iterations=11,
                  maximum_iterations=10000,
                  rtol=1e-10,
                  atol=1e-10,
                  z_maximum_iterations=10000)


# Tests --------------------------------------------------------------------------------------------


@hypothesis.settings(max_examples=100, deadline=5000.)  # type: ignore
@hypothesis.given(
    hypothesis.extra.numpy.arrays(
        np.float_, (5, 5), elements=hypothesis.strategies.floats(0, 1)),
    hypothesis.extra.numpy.arrays(
        np.float_, 5, elements=hypothesis.strategies.floats(0, 1)))
def test_simple_contraction(ax_plus_b: IteratedFunction[TPair, Array, Array, Array, Array],
                            matrix: np.ndarray,
                            offset: np.ndarray) -> None:
    matrix = make_stable(matrix, eps=1e-1)
    x0 = jnp.zeros_like(offset)

    true_sol = solve_ax_b(matrix, offset)
    sol = ax_plus_b.find_fixed_point((matrix, offset), x0).current_state

    assert_allclose(sol, true_sol, rtol=1e-5, atol=1e-5)


@hypothesis.settings(max_examples=100, deadline=5000.)  # type: ignore
@hypothesis.given(
    hypothesis.extra.numpy.arrays(
        np.float_, (5, 5), elements=hypothesis.strategies.floats(0.1, 1)),
    hypothesis.extra.numpy.arrays(
        np.float_, 5, elements=hypothesis.strategies.floats(0.1, 1)))
def test_jvp(ax_plus_b: IteratedFunction[TPair, Array, Array, Array, Array],
             matrix: np.ndarray,
             offset: np.ndarray) -> None:
    matrix = make_stable(matrix, eps=1e-1)
    x0 = jnp.zeros_like(offset)

    def f(theta: TPair, x_init: Array) -> Array:
        return ax_plus_b.find_fixed_point(theta, x_init).current_state

    def f_vjp(theta: TPair, x_init: Array) -> Callable[[Array], TPair]:
        return jax.vjp(f, theta, x_init)  # type: ignore

    jax.test_util.check_vjp(f, f_vjp, ((matrix, offset), x0),  # type: ignore
                            rtol=1e-4, atol=1e-4)


def test_gradient(generator: Generator,
                  ax_plus_b: IteratedFunction[TPair, Array, Array, Array, Array]) -> None:
    """
    Test gradient on the fixed point of Ax + b = x.
    """
    mat_size = 10
    matrix = generate_stable_matrix(generator, mat_size, 1e-1)
    offset = np.random.rand(mat_size)  # type: ignore
    x0 = jnp.zeros_like(offset)

    def loss(params: TPair, x: Array) -> Array:
        return jnp.sum(ax_plus_b.find_fixed_point(params, x).current_state)

    jax.test_util.check_grads(loss,  # type: ignore
                              ((matrix, offset), x0),
                              order=1,
                              modes=["rev"],
                              atol=1e-4,
                              rtol=1e-4)

    grad_matrix, grad_offset = jax.grad(loss)((matrix, offset), x0)

    true_grad_matrix, true_grad_offset = solve_grad_ax_b(matrix, offset)

    assert_allclose(grad_matrix, true_grad_matrix, rtol=1e-4, atol=1e-5)
    assert_allclose(grad_offset, true_grad_offset, rtol=1e-4, atol=1e-5)
