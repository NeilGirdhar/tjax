from __future__ import annotations

from collections.abc import Callable
from typing import override

import jax.numpy as jnp
from jax import tree

from tjax._src.annotations import ComplexNumeric, JaxComplexArray, PyTree, RealNumeric
from tjax.dataclasses import dataclass

from .transform import GradientState, HvpGradientTransformation


@dataclass
class SMDState[Weights: PyTree](GradientState):
    """Gradient state for :class:`SMDGradient`.

    Args:
        log_learning_rate: The logarithm of each weight's learning rate.
        v: The running estimate of the derivative of the weights with respect to the log
            learning rates.
    """

    log_learning_rate: Weights
    v: Weights


@dataclass
class SMDGradient[Weights: PyTree](HvpGradientTransformation[SMDState[Weights], Weights]):
    """Stochastic meta-descent.

    References:
        Schraudolph, 1999,
            [Local gain adaptation in stochastic gradient descent](https://doi.org/10.1049/cp:19991170)

    Args:
        meta_learning_rate: The step size for adapting the log learning rates.
    """

    meta_learning_rate: RealNumeric = 1e-2

    @override
    def init(self, parameters: Weights) -> SMDState[Weights]:
        z = tree.map(jnp.zeros_like, parameters)
        return SMDState[Weights](z, z)

    @override
    def hvp_update(
        self,
        gradient: Weights,
        state: SMDState[Weights],
        parameters: Weights | None,
        hessian_vector_product: Callable[[Weights], Weights],
        **extra_args: object,
    ) -> tuple[Weights, SMDState[Weights]]:
        negative_gradient = tree.map(jnp.negative, gradient)  # delta

        # Update log-learning rate.
        def g(log_p: RealNumeric, delta: ComplexNumeric, v: ComplexNumeric) -> JaxComplexArray:
            return jnp.asarray(log_p + self.meta_learning_rate * delta * v)

        new_log_learning_rate = tree.map(g, state.log_learning_rate, negative_gradient, state.v)
        learning_rate = tree.map(jnp.exp, new_log_learning_rate)  # p

        # Calculate gradient.
        gradient = tree.map(jnp.multiply, learning_rate, negative_gradient)

        # Update v.
        def f(
            v: ComplexNumeric, p: RealNumeric, delta: ComplexNumeric, hv: ComplexNumeric
        ) -> JaxComplexArray:
            return jnp.asarray(v + p * delta - hv)

        new_v = tree.map(
            f, state.v, learning_rate, negative_gradient, hessian_vector_product(state.v)
        )

        return gradient, SMDState[Weights](new_log_learning_rate, new_v)
