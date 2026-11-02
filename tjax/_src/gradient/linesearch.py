"""Linesearches that choose the step size."""

from __future__ import annotations

from typing import override

import optax
from optax import (
    GradientTransformation as OptaxGradientTransformation,
)

from tjax._src.annotations import (
    PyTree,
    RealNumeric,
)
from tjax.dataclasses import as_shallow_dict, dataclass, field

from .transform import (
    OptaxWrapper,
)


@dataclass
class ScaleByBacktrackingLinesearch[Weights: PyTree](OptaxWrapper[Weights]):
    """Backtracking line-search ensuring sufficient decrease (Armijo criterion).

    References:
        Vaswani et al., `Painless Stochastic Gradient <https://arxiv.org/abs/1905.09997>`_, 2019
        Nocedal & Wright, `Numerical Optimization <https://doi.org/10.1007/978-0-387-40065-5>`_,
            1999

    Args:
        max_backtracking_steps: Maximum number of iterations for the line-search.
        slope_rtol: Relative tolerance w.r.t. to the slope. The sufficient decrease must be
            slope_rtol * lr * <grad, updates>, see formula above.
        decrease_factor: Decreasing factor to reduce learning rate.
        increase_factor: Increasing factor to increase learning rate guess. Setting it to 1. amounts
            to keep the current guess, setting it to ``math.inf`` amounts to start with
            ``max_learning_rate`` at each round.
        max_learning_rate: Maximum learning rate (learning rate guess clipped to this).
        atol: Absolute tolerance at which the criterion needs to be satisfied.
        rtol: Relative tolerance at which the criterion needs to be satisfied.
        store_grad: Whether to compute and store the gradient at the end of the linesearch. Since
            the function is called to compute the value to accept the learning rate, we can also
            access the gradient along the way. By doing that, we can directly reuse the value and
            the gradient computed at the end of the linesearch for the next iteration using
            :func:`optax.value_and_grad_from_state`. See the example above.
        verbose: Whether to print debugging information.
    """

    max_backtracking_steps: RealNumeric
    slope_rtol: RealNumeric = 0.0001
    decrease_factor: RealNumeric = 0.8
    increase_factor: RealNumeric = 1.5
    max_learning_rate: RealNumeric = 1.0
    atol: RealNumeric = 0.0
    rtol: RealNumeric = 0.0
    store_grad: bool = field(default=False, static=True)
    verbose: bool = field(default=False, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.scale_by_backtracking_linesearch(**as_shallow_dict(self))


@dataclass
class ScaleByZoomLinesearch[Weights: PyTree](OptaxWrapper[Weights]):
    """Linesearch ensuring sufficient decrease and small curvature.

    References:
        Algorithms 3.5 3.6 of Nocedal and Wright, `Numerical Optimization
            <https://doi.org/10.1007/978-0-387-40065-5>`_, 1999
        Hager and Zhang `Algorithm 851: CG_DESCENT, a Conjugate Gradient Method with Guaranteed
            Descent <https://doi.org/10.1145/1132973.1132979>`_, 2006

    Args:
        max_linesearch_steps: Maximum number of linesearch iterations.
        max_learning_rate: Maximum admissible learning rate. Can be set to ``None`` for no upper
            bound. A non ``None`` value may prevent the linesearch to find a learning rate
            satisfying the small curvature criterion, since the latter may require sufficiently
            large stepsizes.
        tol: Tolerance on the criterions.
        increase_factor: Increasing factor to augment the learning rate when searching for a valid
            interval containing a learning rate satisfying both criterions.
        slope_rtol: Relative tolerance for the slope in the sufficient decrease criterion.
        curv_rtol: Relative tolerance for the curvature in the small curvature criterion.
        approx_dec_rtol: Relative tolerance for the initial value in the approximate sufficient
            decrease criterion. Can be set to ``None`` to use only the original Armijo-Goldstein
            decrease criterion.
        stepsize_precision: Precision in the search of a stepsize satisfying both conditions. The
            algorithm proceeds with a bisection that refines an interval containing a stepsize
            satisfying both conditions. If that interval is reduced below ``stepsize_precision`` and
            a stepsize satisfying a sufficient decrease has been found, the algorithm selects that
            stepsize even if the curvature condition is not satisfied.
        initial_guess_strategy: Initial guess for the learning rate used to start the linesearch.
            Can be either ``one`` or ``keep``. If ``one``, the initial guess is set to 1. If
            ``keep``, the initial guess is set to the learning rate of the previous step. We
            recommend to use ``keep`` if this linesearch is used in combination with SGD. We
            recommend to use ``one`` if this linesearch is used in combination with Newton methods
            or quasi-Newton methods such as L-BFGS.
        verbose: Whether to print additional debugging information in case the linesearch fails.
    """

    max_linesearch_steps: RealNumeric
    max_learning_rate: RealNumeric | None = None
    tol: RealNumeric = 0.0
    increase_factor: RealNumeric = 2.0
    slope_rtol: RealNumeric = 0.0001
    curv_rtol: RealNumeric = 0.9
    approx_dec_rtol: RealNumeric | None = 1e-06
    stepsize_precision: RealNumeric = 1e-05
    initial_guess_strategy: str = field(default="keep", static=True)
    verbose: bool = field(default=False, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.scale_by_zoom_linesearch(**as_shallow_dict(self))
