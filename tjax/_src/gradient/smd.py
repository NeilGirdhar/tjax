from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

import jax.numpy as jnp
from jax import tree
from typing_extensions import override

from tjax.dataclasses import dataclass

from ..annotations import ComplexNumeric, JaxComplexArray, PyTree, RealNumeric
from .transform import GradientState, SecondOrderGradientTransformation

Weights = TypeVar('Weights', bound=PyTree)


@dataclass
class SMDState(GradientState, Generic[Weights]):
    log_learning_rate: Weights
    v: Weights


@dataclass
class SMDGradient(SecondOrderGradientTransformation[SMDState[Weights], Weights],
                  Generic[Weights]):
    """Stochastic meta-descent.

    Schraudolph, N. N. (1999). Local gain adaptation in stochastic gradient descent. Artificial
    Neural Networks, 1999. ICANN 99. Ninth International Conference on (Conf. Publ. No. 470), 2,
    569-574. https://doi.org/10.1049/cp:19991170.
    """
    meta_learning_rate: RealNumeric = 1e-2

    @override
    def init(self, parameters: Weights) -> SMDState[Weights]:
        z = tree.map(jnp.zeros_like, parameters)
        return SMDState[Weights](z, z)

    @override
    def second_order_update(self,
                            gradient: Weights,
                            state: SMDState[Weights],
                            parameters: Weights | None,
                            hessian_vector_product: Callable[[Weights], Weights]
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
        def f(v: ComplexNumeric,
              p: RealNumeric,
              delta: ComplexNumeric,
              hv: ComplexNumeric) -> JaxComplexArray:
            return jnp.asarray(v + p * delta - hv)

        new_v = tree.map(f, state.v, learning_rate, negative_gradient,
                         hessian_vector_product(state.v))

        return gradient, SMDState[Weights](new_log_learning_rate, new_v)

# got
# Union[ndarray[Any, dtype[complexfloating[Any, Any]]],
#       ndarray[Any, dtype[floating[Any]]],
#       ndarray[Any, dtype[signedinteger[Any]]],
#       Array,
#       complex]
# , expected
# Union[ndarray[Any, dtype[Union[floating[Any], complexfloating[Any, Any]]]],
#       Array,
#       complex, float, int]
