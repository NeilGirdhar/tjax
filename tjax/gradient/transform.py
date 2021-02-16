from typing import Callable, Generic, Optional, Tuple, TypeVar

from jax import numpy as jnp
from jax.tree_util import tree_map, tree_multimap, tree_reduce

from ..annotations import PyTree
from ..dataclass import dataclass

__all__ = ['GradientState', 'GradientTransformation', 'SecondOrderGradientTransformation']


GradientState = PyTree
Weights = TypeVar('Weights', bound=PyTree)
State = TypeVar('State', bound=PyTree)


@dataclass
class GradientTransformation(Generic[State, Weights]):
    def init(self, parameters: Weights) -> State:
        raise NotImplementedError

    def update(self,
               gradient: Weights,
               state: State,
               parameters: Optional[Weights]) -> Tuple[Weights, State]:
        """
        Args:
            gradient: The derivative of the loss with respect to the weights.
            state: The gradient state.
            parameters: The weights.
        Returns:
            new_gradient: The modified gradient.
            new_gradient_state: The new gradient state.
        """
        raise NotImplementedError


@dataclass
class SecondOrderGradientTransformation(GradientTransformation[State, Weights],
                                        Generic[State, Weights]):
    def init(self, parameters: Weights) -> State:
        raise NotImplementedError

    def update(self,
               gradient: Weights,
               state: State,
               parameters: Optional[Weights]) -> Tuple[Weights, State]:
        """
        Args:
            gradient: The derivative of the loss with respect to the weights.
            state: The gradient state.
            parameters: The weights.
        Returns:
            new_gradient: The modified gradient.
            new_gradient_state: The new gradient state.
        """
        def hessian_vector_product(v: Weights) -> Weights:
            d = tree_reduce(jnp.add, tree_multimap(jnp.vdot, gradient, v), 0.0)
            return tree_map(lambda x: x * d, gradient)

        return self.second_order_update(gradient, state, parameters, hessian_vector_product)

    def second_order_update(self,
                            gradient: Weights,
                            state: State,
                            parameters: Optional[Weights],
                            hessian_vector_product: Callable[[Weights], Weights]) -> (
                                Tuple[Weights, State]):
        """
        Args:
            gradient: The derivative of the loss with respect to the weights.
            state: The gradient state.
            parameters: The weights.
            hessian_vector_product: A function that maps v to the Hessian of the loss with respect
                ot the weights times v.
        Returns:
            new_gradient: The modified gradient.
            new_gradient_state: The new gradient state.
        """
        raise NotImplementedError
