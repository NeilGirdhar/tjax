from __future__ import annotations

from typing import Any, Optional, Tuple

import jax.numpy as jnp
import pytest
from jax import grad, jit
from numpy.testing import assert_allclose

from tjax import Generator, RealNumeric, dataclass
from tjax.dataclasses import field
from tjax.fixed_point import StochasticIteratedFunctionWithCombinator


@dataclass
class EncodingConfiguration:
    x: RealNumeric
    y: int = field(static=True)


@dataclass
class EncodingState:
    ec: EncodingConfiguration
    rng: Generator


@dataclass
class EncodingIteratedFunction(StochasticIteratedFunctionWithCombinator['EncodingElement',
                                                                        EncodingState,
                                                                        EncodingConfiguration,
                                                                        EncodingConfiguration,
                                                                        Any]):
    time_step: RealNumeric

    # Implemented methods --------------------------------------------------------------------------
    def expected_state(self, theta: EncodingElement, state: EncodingState) -> EncodingState:
        assert isinstance(state, EncodingState)
        new_ec, _ = theta.iterate(state.ec, None, self.time_step)
        return EncodingState(new_ec, state.rng)

    def sampled_state(self, theta: EncodingElement, state: EncodingState) -> EncodingState:
        assert isinstance(state, EncodingState)
        new_ec, new_rng = theta.iterate(state.ec, state.rng, self.time_step)
        assert new_rng is not None
        return EncodingState(new_ec, new_rng)

    def extract_comparand(self, state: EncodingState) -> EncodingConfiguration:
        assert isinstance(state, EncodingState)
        return state.ec

    def extract_differentiand(self, state: EncodingState) -> EncodingConfiguration:
        assert isinstance(state, EncodingState)
        return state.ec

    def implant_differentiand(self,
                              state: EncodingState,
                              differentiand: EncodingConfiguration) -> EncodingState:
        assert isinstance(state, EncodingState)
        assert isinstance(differentiand, EncodingConfiguration)
        return EncodingState(differentiand, state.rng)


@dataclass
class EncodingElement:
    theta: RealNumeric
    diffusion: float = 0.01

    def _initial_state(self) -> EncodingState:
        return EncodingState(EncodingConfiguration(8.0, 1),
                             Generator.from_seed(123))

    def iterate(self,
                ec: EncodingConfiguration,
                rng: Optional[Generator],
                time_step: RealNumeric) -> Tuple[EncodingConfiguration,
                                                 Optional[Generator]]:
        decay = 1e-4
        noise: RealNumeric
        if rng is None:
            new_rng = None
            noise = 0.0
        else:
            noise, new_rng = rng.normal(jnp.sqrt(2.0 * self.diffusion * time_step), shape=())
        x = (ec.x * jnp.exp(-decay * time_step)
             + 10. * (self.theta - ec.x) * time_step
             + noise)
        return EncodingConfiguration(x, ec.y), new_rng

    @jit
    def infer_state(self, eif: EncodingIteratedFunction) -> EncodingState:
        augmented = eif.find_fixed_point(self, self._initial_state())
        return augmented.current_state

    def theta_bar(self, eif: EncodingIteratedFunction) -> RealNumeric:
        def f(encoding: EncodingElement) -> RealNumeric:
            configuration = encoding.infer_state(eif).ec
            return configuration.x
        return grad(f)(self).theta


@pytest.mark.parametrize('theta', (-5.0, -1.0, 0.0, 1.0, 5.0))
def test_use_classes(theta: float) -> None:
    eif = EncodingIteratedFunction(minimum_iterations=11,
                                   maximum_iterations=1000,
                                   atol=1e-2,
                                   convergence_detection_decay=0.1,
                                   time_step=0.01)

    encoding = EncodingElement(theta)

    assert_allclose(theta,
                    encoding.infer_state(eif).ec.x,
                    rtol=1e-1,
                    atol=1e-1)
    assert_allclose(1.0,
                    encoding.theta_bar(eif),
                    atol=1e-2)
