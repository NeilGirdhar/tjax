from __future__ import annotations

from functools import partial
from typing import Generic, Tuple

from jax import numpy as jnp
from jax import vjp
from jax.tree_util import tree_multimap

from tjax import custom_vjp, dataclass

from .augmented import State
from .comparing import ComparingIteratedFunction
from .iterated_function import IteratedFunction, Parameters, TheAugmentedState

__all__ = ['IteratedFunctionWithCombinator']


@dataclass
class _ZResiduals(Generic[Parameters, State, TheAugmentedState]):
    outer_iterated_function: IteratedFunctionWithCombinator[Parameters, State, TheAugmentedState]
    outer_theta: Parameters
    x_star: State


@dataclass
class _ZParameters(Generic[Parameters, State]):
    outer_theta: Parameters
    x_star: State
    x_star_bar: State


@dataclass
class _ZIterate(ComparingIteratedFunction[_ZParameters[Parameters, State], State, State],
                Generic[Parameters, State, TheAugmentedState]):

    iterated_function: IteratedFunction[Parameters, State, TheAugmentedState]

    # Implemented methods --------------------------------------------------------------------------
    def iterate_state(self, theta: _ZParameters[Parameters, State], x: State) -> State:
        # The state should be called z, but we can't change the interface because of Liskov's
        # substitution principle.
        z = x
        del x

        def f_of_x(x: State) -> State:
            return self.iterated_function.iterate_state(theta.outer_theta, x)

        _, df_by_dx = vjp(f_of_x, theta.x_star)
        df_by_dx_times_z, = df_by_dx(z)
        return tree_multimap(jnp.add, theta.x_star_bar, df_by_dx_times_z)

    def extract_comparand(self, state: State) -> State:
        return state


def _ffp_fwd(outer_iterated_function: IteratedFunctionWithCombinator[Parameters, State,
                                                                     TheAugmentedState],
             theta: Parameters,
             initial_state: State) -> Tuple[TheAugmentedState, _ZResiduals[Parameters, State,
                                                                           TheAugmentedState]]:
    """
    Args:
        theta: The parameters for which gradients can be calculated.
        initial_state: An initial guess of the final state.
    Returns:
        x_star: the result of the minimization.
        residuals: residuals used in _ffp_bwd.
    """
    augmented: TheAugmentedState = outer_iterated_function.find_fixed_point(
        theta, initial_state)
    return augmented, _ZResiduals(outer_iterated_function, theta, augmented.current_state)


def _ffp_bwd(residuals: _ZResiduals[Parameters, State, TheAugmentedState],
             augmented_star_bar: TheAugmentedState) -> Tuple[Parameters]:
    """
    Args:
        residuals: residuals produced by _ffp_fwd.
        augmented_star_bar: cotangents
    Returns:
        theta_bar: cotangents for theta
        zeroed_xs: cotangents for initial_state
    """
    # pylint: disable=protected-access
    outer_iterated_function = residuals.outer_iterated_function
    x_star_bar = augmented_star_bar.current_state

    def f_of_theta(some_theta: Parameters) -> State:
        return outer_iterated_function.iterate_state(some_theta, residuals.x_star)

    z_iterator = _ZIterate(iteration_limit=outer_iterated_function.z_iteration_limit,
                           iterated_function=outer_iterated_function)
    z_parameters = _ZParameters(residuals.outer_theta, residuals.x_star, x_star_bar)
    z_star: State = z_iterator.find_fixed_point(z_parameters, x_star_bar).current_state

    _, df_by_dtheta = vjp(f_of_theta, residuals.outer_theta)
    theta_bar, = df_by_dtheta(z_star)
    return (theta_bar,)


# https://github.com/python/mypy/issues/8539
@dataclass  # type: ignore
class IteratedFunctionWithCombinator(IteratedFunction[Parameters, State, TheAugmentedState],
                                     Generic[Parameters, State, TheAugmentedState]):
    """
    An IteratedFunctionWithCombinator is an IteratedFunction that invokes a combinator so that
    differentiation works through the fixed point.  Besides inheriting from this class, no other
    action is necessary to get this capability.

    Attributes:
        z_iteration_limit:
            The maximum number of iterations to use to evaluate the adjoint's fixed point.
    """
    z_iteration_limit: int = 1000

    # Overridden methods ---------------------------------------------------------------------------
    @partial(custom_vjp[TheAugmentedState], nondiff_argnums=(0, 2))
    def find_fixed_point(self,  # type: ignore
                         theta: Parameters,
                         initial_state: State) -> TheAugmentedState:
        """
        Args:
            theta: The parameters for which gradients can be calculated.
            initial_state: An initial guess of the final state.
        Returns: The augmented state at the fixed point.
        """
        return super().find_fixed_point(theta, initial_state)

    # Apply vjp ------------------------------------------------------------------------------------
    find_fixed_point.defvjp(_ffp_fwd, _ffp_bwd)
