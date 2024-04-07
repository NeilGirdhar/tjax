from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import pytest
from jax import grad
from jax.random import key, normal, split
from numpy.testing import assert_allclose
from typing_extensions import override

from tjax import KeyArray, RealNumeric, jit
from tjax.dataclasses import dataclass, field
from tjax.fixed_point import StochasticIteratedFunctionWithCombinator


@dataclass
class EncodingConfiguration:
    x: RealNumeric
    y: int = field(static=True)


@dataclass
class EncodingState:
    ec: EncodingConfiguration
    rng: KeyArray


@dataclass
class EncodingIteratedFunction(StochasticIteratedFunctionWithCombinator['EncodingElement',
                                                                        EncodingState,
                                                                        EncodingConfiguration,
                                                                        EncodingConfiguration,
                                                                        Any]):
    time_step: RealNumeric

    # Implemented methods --------------------------------------------------------------------------
    @override
    def expected_state(self, theta: EncodingElement, state: EncodingState) -> EncodingState:
        assert isinstance(state, EncodingState)
        new_ec, _ = theta.iterate(state.ec, None, self.time_step)
        return EncodingState(new_ec, state.rng)

    @override
    def sampled_state(self, theta: EncodingElement, state: EncodingState) -> EncodingState:
        assert isinstance(state, EncodingState)
        new_ec, new_rng = theta.iterate(state.ec, state.rng, self.time_step)
        assert new_rng is not None
        return EncodingState(new_ec, new_rng)

    @override
    def extract_comparand(self, state: EncodingState) -> EncodingConfiguration:
        assert isinstance(state, EncodingState)
        return state.ec

    @override
    def extract_differentiand(self,
                              theta: EncodingElement,
                              state: EncodingState) -> EncodingConfiguration:
        assert isinstance(state, EncodingState)
        return state.ec

    @override
    def implant_differentiand(self,
                              theta: EncodingElement,
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
        return EncodingState(EncodingConfiguration(8.0, 1), key(123))

    def iterate(self,
                ec: EncodingConfiguration,
                rng: KeyArray | None,
                time_step: RealNumeric) -> tuple[EncodingConfiguration, KeyArray | None]:
        decay = 1e-4
        noise: RealNumeric
        if rng is None:
            new_rng = None
            noise = 0.0
        else:
            normal_rng, new_rng = split(rng)
            noise = jnp.sqrt(2.0 * self.diffusion * time_step) * normal(normal_rng)
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


@pytest.mark.parametrize('theta', [-5.0, -1.0, 0.0, 1.0, 5.0])
def test_use_classes(theta: float) -> None:
    eif = EncodingIteratedFunction(minimum_iterations=11,
                                   maximum_iterations=1000,
                                   rtol=1e-4,
                                   atol=1e-2,
                                   z_minimum_iterations=11,
                                   z_maximum_iterations=1000,
                                   convergence_detection_decay=jnp.asarray(0.1),
                                   time_step=0.01)

    encoding = EncodingElement(theta)

    assert_allclose(theta, encoding.infer_state(eif).ec.x, rtol=1e-1, atol=1e-1)
    assert_allclose(1.0, encoding.theta_bar(eif), atol=1e-2)
