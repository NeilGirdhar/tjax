from __future__ import annotations

from typing import Callable, Generic, Optional, Tuple, Type, TypeVar

import jax.numpy as jnp
from jax.tree_util import tree_map, tree_reduce

from ..annotations import PyTree
from ..dataclasses import dataclass
from ..tools import abs_square

__all__ = ['GradientState', 'GradientTransformation', 'SecondOrderGradientTransformation',
           'ThirdOrderGradientTransformation']


# Gradient states ----------------------------------------------------------------------------------
T = TypeVar('T', bound='GenericGradientState')
U = TypeVar('U', bound=PyTree)


@dataclass
class GradientState:
    """
    This base class allows instance checks, and strict type annotations.
    """


@dataclass
class GenericGradientState(GradientState):
    """
    This class serves as a simple container for gradient states because optax doesn't provide a
    suitable base class.
    """
    data: PyTree

    @classmethod
    def wrap(cls: Type[T], weights: Weights, state: PyTree) -> Tuple[Weights, T]:
        return weights, cls(state)


# The weights type variable must be the same type variable that is used for the gradients.  After
# all, these must be compatible pytrees.
Weights = TypeVar('Weights', bound=PyTree)
State = TypeVar('State', bound=GradientState)


# Gradient transformation base classes -------------------------------------------------------------
@dataclass
class GradientTransformation(Generic[State, Weights]):
    """
    A gradient transformation transforms loss gradients (with respect to weights) into weight deltas
    that are added to the weights.  The typical goal of a gradient transformation is for the weights
    to converge quickly and stably.

    The most basic gradient transformation is the scaled negative gradients, which can be
    interpreted as doing Newton's method with an assumed isotropic loss Hessian.  In general, all
    gradient transformations can be interpreted as Hessian estimators.
    """
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
    """
    A second order gradient transformation is a special case of a gradient transformation whose
    update rule accepts a Hessian-vector-product function in its update.
    """
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

        This uses the outer product approximation of the Hessian.
        """
        def hessian_vector_product(v: Weights) -> Weights:
            d = tree_reduce(jnp.add, tree_map(jnp.vdot, gradient, v), 0.0)
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
                to the weights times v.
        Returns:
            new_gradient: The modified gradient.
            new_gradient_state: The new gradient state.
        """
        raise NotImplementedError


@dataclass
class ThirdOrderGradientTransformation(SecondOrderGradientTransformation[State, Weights],
                                       Generic[State, Weights]):
    """
    A third order gradient transformation is a special case of a second order gradient
    transformation whose update rule accepts the diagonal entries of the Hessian.
    """
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
                to the weights times v.
        Returns:
            new_gradient: The modified gradient.
            new_gradient_state: The new gradient state.

        Uses the outer product approximation of the Hessian to provide the diagonal entries of the
        Hessian.
        """
        return self.third_order_update(gradient, state, parameters, hessian_vector_product,
                                       tree_map(abs_square, gradient))

    def third_order_update(self,
                           gradient: Weights,
                           state: State,
                           parameters: Optional[Weights],
                           hessian_vector_product: Callable[[Weights], Weights],
                           hessian_diagonal: Weights) -> Tuple[Weights, State]:
        """
        Args:
            gradient: The derivative of the loss with respect to the weights.
            state: The gradient state.
            parameters: The weights.
            hessian_vector_product: A function that maps v to the Hessian of the loss with respect
                to the weights times v.
            hessian_diagonal: The diagonal entries of the Hessian of the loss with respect to the
                weights.
        Returns:
            new_gradient: The modified gradient.
            new_gradient_state: The new gradient state.
        """
        raise NotImplementedError
