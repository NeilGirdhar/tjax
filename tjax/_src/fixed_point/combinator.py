from __future__ import annotations

from typing import Any, Generic, TypeVar

import jax.numpy as jnp
from jax import vjp
from jax.tree_util import tree_map
from typing_extensions import override

from ..annotations import PyTree
from ..dataclasses import dataclass
from ..shims import custom_vjp_method
from .augmented import State
from .base import Parameters, Trajectory
from .comparing import ComparingIteratedFunction, ComparingState
from .iterated_function import Comparand, IteratedFunction, TheAugmentedState

__all__ = ['IteratedFunctionWithCombinator', 'ComparingIteratedFunctionWithCombinator']


Differentiand = TypeVar('Differentiand', bound=PyTree)


@dataclass
class _ZResiduals(Generic[Parameters, State, Comparand, Differentiand, TheAugmentedState]):
    outer_iterated_function: IteratedFunctionWithCombinator[Parameters, State, Comparand,
                                                            Differentiand, Any, TheAugmentedState]
    outer_theta: Parameters
    x_star: State


@dataclass
class _ZParameters(Generic[Parameters, State, Differentiand]):
    outer_theta: Parameters
    x_star: State
    x_star_differentiand: Differentiand
    x_star_bar_differentiand: Differentiand


def _ffp_bwd(residuals: _ZResiduals[Parameters, State, Comparand, Differentiand, TheAugmentedState],
             augmented_star_bar: TheAugmentedState) -> tuple[None, Parameters, None]:
    """The backward pass of the fixed point finder.

    Args:
        residuals: residuals produced by _ffp_fwd.
        augmented_star_bar: cotangents
    Returns:
        theta_bar: cotangents for theta
        zeroed_xs: cotangents for initial_state
    """
    outer_iterated_function = residuals.outer_iterated_function
    outer_theta = residuals.outer_theta
    x_star = residuals.x_star
    x_star_differentiand = outer_iterated_function.extract_differentiand(outer_theta, x_star)
    x_star_bar = augmented_star_bar.current_state
    x_star_bar_differentiand = outer_iterated_function.extract_differentiand(outer_theta,
                                                                             x_star_bar)

    def f_of_theta(some_theta: Parameters) -> Differentiand:
        state = outer_iterated_function.expected_state(some_theta, x_star)
        return outer_iterated_function.extract_differentiand(outer_theta, state)

    z_iterator = _ZIterate(minimum_iterations=outer_iterated_function.z_minimum_iterations,
                           maximum_iterations=outer_iterated_function.z_maximum_iterations,
                           rtol=outer_iterated_function.rtol, atol=outer_iterated_function.atol,
                           z_minimum_iterations=outer_iterated_function.z_minimum_iterations,
                           z_maximum_iterations=outer_iterated_function.z_maximum_iterations,
                           iterated_function=outer_iterated_function)
    z_parameters = _ZParameters(residuals.outer_theta, x_star, x_star_differentiand,
                                x_star_bar_differentiand)
    augmented = z_iterator.find_fixed_point(z_parameters, x_star_bar_differentiand)
    z_star_differentiand: Differentiand = augmented.current_state

    _, df_by_dtheta = vjp(f_of_theta, residuals.outer_theta)
    theta_bar, = df_by_dtheta(z_star_differentiand)
    return None, theta_bar, None


@dataclass
class IteratedFunctionWithCombinator(
        IteratedFunction[Parameters, State, Comparand, Trajectory, TheAugmentedState],
        Generic[Parameters, State, Comparand, Differentiand, Trajectory, TheAugmentedState]):
    """An IteratedFunctionWithCombinator is an IteratedFunction that invokes a combinator.

    It allows differentiation works through the fixed point.  Besides inheriting from this class, no
    other action is necessary to get this capability.

    It is a generic class with all of the parameters of IteratedFunction, and Differentiand, which
    is the type of the *portion of the state* with respect to which derivatives at the fixed point
    are calculated.

    Attributes:
        z_maximum_iterations:
            The maximum number of iterations to use to evaluate the adjoint's fixed point.
    """
    z_minimum_iterations: int
    z_maximum_iterations: int

    # Overridden methods ---------------------------------------------------------------------------
    @custom_vjp_method
    @override
    def find_fixed_point(self,  # type: ignore[override] # pyright: ignore
                         theta: Parameters,
                         initial_state: State) -> TheAugmentedState:
        """Find the fixed point.

        This is a separate method because it has a bound custom VJP.

        Args:
            theta: The parameters for which gradients can be calculated.
            initial_state: An initial guess of the final state.
        Returns: The augmented state at the fixed point.
        """
        return super().find_fixed_point(theta, initial_state)

    # Abstract methods -----------------------------------------------------------------------------
    def extract_differentiand(self, theta: Parameters, state: State) -> Differentiand:
        """Extract the differentiable values in a state.

        It is used by the combinator to find cotangents.

        Returns: The differentiable values in the state.
        """
        raise NotImplementedError

    def implant_differentiand(self,
                              theta: Parameters,
                              state: State,
                              differentiand: Differentiand) -> State:
        """Implant the differentiand into the state.

        Args:
            theta: The parameters for which gradients can be calculated.
            state: A state that will provide nondifferentiable values.
            differentiand: A differentiand that will provide differentiable values.
        Returns: A state containing differentiable from the differentiand and nondifferentiable
            values from the inputted state.
        """
        raise NotImplementedError

    # Apply vjp ------------------------------------------------------------------------------------
    def _ffp_fwd(self,
                 theta: Parameters,
                 initial_state: State) -> tuple[TheAugmentedState,
                                                _ZResiduals[Parameters, State, Comparand,
                                                            Differentiand, TheAugmentedState]]:
        """The forward pass of the fixed point finder.

        Args:
            theta: The parameters for which gradients can be calculated.
            initial_state: An initial guess of the final state.

        Returns:
            x_star: the result of the minimization.
            residuals: residuals used in _ffp_bwd.
        """
        augmented: TheAugmentedState = self.find_fixed_point(theta, initial_state)
        return augmented, _ZResiduals(self, theta, augmented.current_state)

    find_fixed_point.defvjp(_ffp_fwd, _ffp_bwd)


class ComparingIteratedFunctionWithCombinator(
        IteratedFunctionWithCombinator[Parameters, State, Comparand, Differentiand, Trajectory,
                                       ComparingState[State, Comparand]],
        ComparingIteratedFunction[Parameters, State, Comparand, Trajectory],
        Generic[Parameters, State, Comparand, Differentiand, Trajectory]):
    pass


@dataclass
class _ZIterate(ComparingIteratedFunctionWithCombinator[
        _ZParameters[Parameters, State, Differentiand],
        Differentiand,
        Differentiand,
        Differentiand,
        None],
        Generic[Parameters, State, Comparand, TheAugmentedState, Differentiand]):
    """The state of _ZIterate is the differentiand of the outer iterated function."""
    iterated_function: IteratedFunctionWithCombinator[
        Parameters, State, Comparand, Differentiand, Any, TheAugmentedState]

    # Implemented methods --------------------------------------------------------------------------
    @override
    def expected_state(self,
                       theta: _ZParameters[Parameters, State, Differentiand],
                       state: Differentiand) -> Differentiand:
        return self.sampled_state(theta, state)

    @override
    def sampled_state(self,
                      theta: _ZParameters[Parameters, State, Differentiand],
                      state: Differentiand) -> Differentiand:
        # The state should be called z, but we can't change the interface because of Liskov's
        # substitution principle.
        z = state
        del state

        def f_of_x(x_differentiand: Differentiand) -> Differentiand:
            x = self.iterated_function.implant_differentiand(theta.outer_theta, theta.x_star,
                                                             x_differentiand)
            state = self.iterated_function.expected_state(theta.outer_theta, x)
            return self.iterated_function.extract_differentiand(theta.outer_theta, state)

        _, df_by_dx = vjp(f_of_x, theta.x_star_differentiand)
        df_by_dx_times_z, = df_by_dx(z)
        return tree_map(jnp.add, theta.x_star_bar_differentiand, df_by_dx_times_z)

    @override
    def extract_comparand(self, state: Differentiand) -> Differentiand:
        return state

    @override
    def extract_differentiand(self,
                              theta: _ZParameters[Parameters, State, Differentiand],
                              state: Differentiand) -> Differentiand:
        return state

    @override
    def implant_differentiand(self,
                              theta: _ZParameters[Parameters, State, Differentiand],
                              state: Differentiand,
                              differentiand: Differentiand) -> Differentiand:
        return differentiand
