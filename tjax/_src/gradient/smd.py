from __future__ import annotations

from typing import Callable, Generic, Optional, Tuple, TypeVar

import jax.numpy as jnp
from jax.tree_util import tree_map

from ..annotations import ComplexNumeric, PyTree, RealNumeric
from ..dataclasses import dataclass
from .transform import GradientState, SecondOrderGradientTransformation

__all__ = ['SMDState', 'SMDGradient']


Weights = TypeVar('Weights', bound=PyTree)


@dataclass
class SMDState(GradientState, Generic[Weights]):
    log_learning_rate: Weights
    v: Weights


@dataclass
class SMDGradient(SecondOrderGradientTransformation[SMDState[Weights], Weights],
                  Generic[Weights]):
    """
    Schraudolph, N. N. (1999). Local gain adaptation in stochastic gradient descent. Artificial
    Neural Networks, 1999. ICANN 99. Ninth International Conference on (Conf. Publ. No. 470), 2,
    569–574. https://doi.org/10.1049/cp:19991170
    """
    meta_learning_rate: RealNumeric = 1e-2

    def init(self, parameters: Weights) -> SMDState[Weights]:
        z = tree_map(jnp.zeros_like, parameters)
        return SMDState[Weights](z, z)

    def second_order_update(self,
                            gradient: Weights,
                            state: SMDState[Weights],
                            parameters: Optional[Weights],
                            hessian_vector_product: Callable[[Weights], Weights]) -> (
                                Tuple[Weights, SMDState[Weights]]):
        negative_gradient = tree_map(jnp.negative, gradient)  # delta

        # Update log-learning rate.
        def g(log_p: RealNumeric, delta: ComplexNumeric, v: ComplexNumeric) -> ComplexNumeric:
            return log_p + self.meta_learning_rate * delta * v

        new_log_learning_rate = tree_map(g, state.log_learning_rate, negative_gradient, state.v)
        learning_rate = tree_map(jnp.exp, new_log_learning_rate)  # p

        # Calculate gradient.
        gradient = tree_map(jnp.multiply, learning_rate, negative_gradient)

        # Update v.
        def f(v: ComplexNumeric,
              p: RealNumeric,
              delta: ComplexNumeric,
              hv: ComplexNumeric) -> ComplexNumeric:
            return v + p * delta - hv  # type: ignore[return-value]

        new_v = tree_map(f, state.v, learning_rate, negative_gradient,
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
