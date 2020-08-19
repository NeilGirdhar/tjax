from __future__ import annotations

from typing import Optional, Tuple, TypeVar

import pytest
from chex import Array
from jax import grad, jit
from jax import numpy as jnp
from numpy.testing import assert_allclose

from tjax import Generator, dataclass, field, real_dtype
from tjax.fixed_point import StochasticIteratedFunctionWithCombinator

T = TypeVar('T', bound='EncodingIteratedFunction')
U = TypeVar('U', bound='EncodingConfiguration')
V = TypeVar('V', bound='EncodingElement')


@dataclass
class EncodingIteratedFunction(StochasticIteratedFunctionWithCombinator['EncodingElement',
                                                                        'EncodingConfiguration',
                                                                        'EncodingConfiguration']):

    time_step: real_dtype

    # Implemented methods --------------------------------------------------------------------------
    def extract_comparand(self, state: EncodingConfiguration) -> EncodingConfiguration:
        return state

    def iterate_state(self,
                      theta: EncodingElement,
                      x: EncodingConfiguration) -> EncodingConfiguration:
        new_ec, _ = theta.iterate(x, None, self.time_step)
        return new_ec

    def stochastic_iterate_state(self,
                                 theta: EncodingElement,
                                 state: EncodingConfiguration,
                                 rng: Generator) -> Tuple[EncodingConfiguration, Generator]:
        return theta.iterate(state, rng, self.time_step)


@dataclass
class EncodingConfiguration:
    x: Array
    y: int = field(static=True)


@dataclass
class EncodingElement:

    theta: Array
    diffusion: float = 0.01

    def _initial_configuration(self) -> EncodingConfiguration:
        return EncodingConfiguration(8.0, 1)

    def iterate(self,
                ec: EncodingConfiguration,
                rng: Optional[Generator],
                time_step: real_dtype) -> Tuple[EncodingConfiguration,
                                                Optional[Generator]]:
        decay = 1e-4
        if rng is None:
            new_rng = None
            noise = 0.0
        else:
            new_rng, noise = rng.normal(
                jnp.sqrt(2.0 * self.diffusion * time_step), shape=())
        x = (ec.x * jnp.exp(-decay * time_step)
             + 10. * (self.theta - ec.x) * time_step
             + noise)
        return EncodingConfiguration(x, ec.y), new_rng

    @jit
    def infer_configuration(self, eif: EncodingIteratedFunction) -> (
            EncodingConfiguration):
        augmented = eif.find_fixed_point(self, self._initial_configuration())
        return augmented.current_state

    def theta_bar(self, eif: EncodingIteratedFunction) -> Array:
        def f(encoding: EncodingElement) -> EncodingConfiguration:
            configuration = encoding.infer_configuration(eif)
            return configuration.x
        return grad(f)(self).theta


@pytest.mark.parametrize('theta', (-5.0, -1.0, 0.0, 1.0, 5.0))
def test_use_classes(theta: float) -> None:
    eif = EncodingIteratedFunction(iteration_limit=1000,
                                   atol=1e-2,
                                   initial_rng=Generator(seed=123),
                                   decay=0.1,
                                   time_step=0.01)

    encoding = EncodingElement(theta)

    assert_allclose(theta,
                    encoding.infer_configuration(eif).x,
                    rtol=1e-1,
                    atol=1e-1)
    assert_allclose(1.0,
                    encoding.theta_bar(eif),
                    atol=1e-2)
