from __future__ import annotations

from collections.abc import Callable
from typing import Self, override

import jax.numpy as jnp
from jax import tree

from tjax._src.annotations import PyTree
from tjax._src.math_tools import abs_square
from tjax.dataclasses import dataclass


# Gradient states ----------------------------------------------------------------------------------
@dataclass
class GradientState:
    """This base class allows instance checks, and strict type annotations."""


@dataclass
class GenericGradientState(GradientState):
    """A simple container for gradient states.

    Optax doesn't provide a suitable base class.
    """

    data: PyTree

    @classmethod
    def wrap[Weights: PyTree](cls, weights: Weights, state: PyTree) -> tuple[Weights, Self]:
        return weights, cls(state)


# Gradient transformation base classes -------------------------------------------------------------
# The weights type variable must be the same type variable that is used for the gradients.  After
# all, these must be compatible pytrees.
@dataclass
class GradientTransformation[State: GradientState, Weights: PyTree]:
    """A class that transforms weight cotangents into into weight deltas.

    The delta are added to the weights.  The typical goal of a gradient transformation is for the
    weights to converge quickly and stably.

    The most basic gradient transformation is the scaled negative gradients, which can be
    interpreted as doing Newton's method with an assumed isotropic loss Hessian.  In general, all
    gradient transformations can be interpreted as Hessian estimators.
    """

    def init(self, parameters: Weights) -> State:
        raise NotImplementedError

    def update(
        self, gradient: Weights, state: State, parameters: Weights | None
    ) -> tuple[Weights, State]:
        """Transform the weight gradient and update the gradient state.

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
class HvpGradientTransformation[State: GradientState, Weights: PyTree](
    GradientTransformation[State, Weights]
):
    """A gradient transformation whose update rule accepts a Hessian-vector-product function.

    Subclasses implement :meth:`hvp_update`, which receives a callable ``hvp`` that maps any
    vector ``v`` to the Hessian-of-loss times ``v``.

    The default :meth:`update` supplies ``hvp`` using the rank-1 outer-product approximation
    ``H ≈ g gᵀ``, which is exact only when the gradient is a rank-1 vector.  Callers that have
    access to a true HVP (e.g. via :func:`jax.linear_util.wrap_init` or forward-over-reverse AD)
    should call :meth:`hvp_update` directly.
    """

    @override
    def update(
        self, gradient: Weights, state: State, parameters: Weights | None
    ) -> tuple[Weights, State]:
        """Transform the weight gradient and update the gradient state.

        Args:
            gradient: The derivative of the loss with respect to the weights.
            state: The gradient state.
            parameters: The weights.

        Returns:
            new_gradient: The modified gradient.
            new_gradient_state: The new gradient state.

        Note:
            Approximates the Hessian-vector product using the rank-1 outer-product
            ``H v ≈ (gᵀ v) g``.  This is exact only when the loss Hessian is rank-1
            (i.e. the gradient is a scalar multiple of a fixed direction).  For a true
            HVP, call :meth:`hvp_update` directly.
        """

        def hessian_vector_product(v: Weights) -> Weights:
            d = tree.reduce_associative(jnp.add, tree.map(jnp.vdot, gradient, v), identity=0.0)  # type: ignore
            return tree.map(lambda x: x * d, gradient)

        return self.hvp_update(gradient, state, parameters, hessian_vector_product)

    def hvp_update(
        self,
        gradient: Weights,
        state: State,
        parameters: Weights | None,
        hessian_vector_product: Callable[[Weights], Weights],
    ) -> tuple[Weights, State]:
        """Transform the weight gradient and update the gradient state.

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
class DiagHessianGradientTransformation[State: GradientState, Weights: PyTree](
    HvpGradientTransformation[State, Weights]
):
    """A gradient transformation whose update rule accepts a diagonal Hessian estimate.

    Subclasses implement :meth:`diag_hessian_update`, which additionally receives the
    element-wise diagonal of the Hessian.

    The default :meth:`hvp_update` supplies the diagonal using the rank-1 outer-product
    approximation ``diag(H) ≈ g²`` (element-wise square of the gradient), which is exact
    only when the gradient is a rank-1 vector.
    """

    @override
    def hvp_update(
        self,
        gradient: Weights,
        state: State,
        parameters: Weights | None,
        hessian_vector_product: Callable[[Weights], Weights],
    ) -> tuple[Weights, State]:
        """Transform the weight gradient and update the gradient state.

        Args:
            gradient: The derivative of the loss with respect to the weights.
            state: The gradient state.
            parameters: The weights.
            hessian_vector_product: A function that maps v to the Hessian of the loss with respect
                to the weights times v.

        Returns:
            new_gradient: The modified gradient.
            new_gradient_state: The new gradient state.

        Note:
            Approximates the Hessian diagonal using the rank-1 outer-product
            ``diag(H) ≈ g²`` (element-wise square of the gradient).  This is exact only
            when the loss Hessian is rank-1.
        """
        return self.diag_hessian_update(
            gradient, state, parameters, hessian_vector_product, tree.map(abs_square, gradient)
        )

    def diag_hessian_update(
        self,
        gradient: Weights,
        state: State,
        parameters: Weights | None,
        hessian_vector_product: Callable[[Weights], Weights],
        hessian_diagonal: Weights,
    ) -> tuple[Weights, State]:
        """Transform the weight gradient and update the gradient state.

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
