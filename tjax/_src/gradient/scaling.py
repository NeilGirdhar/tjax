"""Transformations that rescale updates."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import KW_ONLY
from typing import Any, Literal, override

import jax.numpy as jnp
import optax
from optax import (
    GradientTransformation as OptaxGradientTransformation,
)

from tjax._src.annotations import (
    PyTree,
    RealNumeric,
)
from tjax.dataclasses import as_shallow_dict, dataclass, field

from .schedule import ScalarOrSchedule, Schedule
from .transform import (
    OptaxWrapper,
)


def _decay_rate_pow(i: RealNumeric, exponent: RealNumeric = 0.8) -> RealNumeric:
    """Second-moment decay rate, as in optax's ``scale_by_factored_rms``."""
    t = jnp.array(i + 1, jnp.float32)
    return 1.0 - t ** (-exponent)


@dataclass
class Scale[Weights: PyTree](OptaxWrapper[Weights]):
    """Scale updates by some fixed scalar `step_size`.

    Args:
        step_size: a scalar corresponding to a fixed scaling factor for updates.
    """

    step_size: RealNumeric

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.scale(**as_shallow_dict(self))


@dataclass
class ScaleBySchedule[Weights: PyTree](OptaxWrapper[Weights]):
    """Scale updates using a custom schedule for the `step_size`.

    Args:
        step_size_fn: a function that takes an update count as input and proposes
            the step_size to multiply the updates by.
    """

    step_size_fn: Schedule

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.scale_by_schedule(**as_shallow_dict(self))


@dataclass
class ScaleByLearningRate[Weights: PyTree](OptaxWrapper[Weights]):
    """Scale by the (negative) learning rate (either as scalar or as schedule).

    Args:
        learning_rate: Can either be a scalar or a schedule (i.e. a callable that maps an (int) step
            to a float). None means no scaling.
        flip_sign: When set to True (the default) this corresponds to scaling by the negative
            learning rate.
    """

    learning_rate: ScalarOrSchedule | None = None
    _: KW_ONLY
    flip_sign: bool = field(default=True, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.scale_by_learning_rate(**as_shallow_dict(self))


@dataclass
class ScaleByAdaDelta[Weights: PyTree](OptaxWrapper[Weights]):
    """Rescale updates according to the Adadelta algorithm.

    Args:
        rho: A coefficient used for computing a running average of squared gradients.
        eps: Term added to the denominator to improve numerical stability.
    """

    rho: RealNumeric = 0.9
    eps: RealNumeric = 1e-06

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.scale_by_adadelta(**as_shallow_dict(self))


@dataclass
class ScaleByAdam[Weights: PyTree](OptaxWrapper[Weights]):
    """Rescale updates according to the Adam algorithm (without a learning rate).

    References:
        Kingma and Ba, 2014,
            [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)

    Args:
        b1: Decay rate for the first moment of gradients.
        b2: Decay rate for the second moment of gradients.
        eps: Regularisation term for numerical stability.
        eps_root: Term added inside the square-root for numerical stability.
        mu_dtype: Optional dtype for the first moment accumulator.
        nesterov: Whether to use Nesterov momentum.
    """

    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.999
    eps: RealNumeric = 1e-8
    eps_root: RealNumeric = 0.0
    mu_dtype: Any = field(default=None, static=True)
    _: KW_ONLY
    nesterov: bool = field(default=False, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.scale_by_adam(**as_shallow_dict(self))


@dataclass
class ScaleByAdamax[Weights: PyTree](OptaxWrapper[Weights]):
    """Rescale updates according to the Adamax algorithm.

    Args:
        b1: Decay rate for the exponentially weighted average of grads.
        b2: Decay rate for the exponentially weighted maximum of grads.
        eps: Term added to the denominator to improve numerical stability.
    """

    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.999
    eps: RealNumeric = 1e-08

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.scale_by_adamax(**as_shallow_dict(self))


@dataclass
class ScaleByAdan[Weights: PyTree](OptaxWrapper[Weights]):
    """Rescale updates according to the Adan algorithm.

    Args:
        b1: Decay rate for the EWMA of gradients.
        b2: Decay rate for the EWMA of differences of gradients.
        b3: Decay rate for the EMWA of the algorithm's squared term.
        eps: Term added to the denominator to improve numerical stability.
        eps_root: Term added to the denominator inside the square-root to improve numerical
            stability when backpropagating gradients through the rescaling.
    """

    b1: RealNumeric = 0.98
    b2: RealNumeric = 0.92
    b3: RealNumeric = 0.99
    eps: RealNumeric = 1e-08
    eps_root: RealNumeric = 0.0

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.scale_by_adan(**as_shallow_dict(self))


@dataclass
class ScaleByAMSGrad[Weights: PyTree](OptaxWrapper[Weights]):
    """Rescale updates according to the AMSGrad algorithm.

    Args:
        b1: Decay rate for the exponentially weighted average of grads.
        b2: Decay rate for the exponentially weighted average of squared grads.
        eps: Term added to the denominator to improve numerical stability.
        eps_root: Term added to the denominator inside the square-root to improve numerical
            stability when backpropagating gradients through the rescaling.
        mu_dtype: Optional `dtype` to be used for the first order accumulator; if `None` then the
            `dtype` is inferred from `params` and `updates`.
        bias_correction_mu: Whether to apply bias correction to the first moment estimate. Set to
            ``False`` to match the original AMSGrad paper.
        bias_correction_nu: Whether to apply bias correction to the second moment estimate before
            taking the elementwise maximum (``nu_max``). Set to ``False`` to match the original
            AMSGrad paper.
    """

    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.999
    eps: RealNumeric = 1e-08
    eps_root: RealNumeric = 0.0
    mu_dtype: Any = field(default=None, static=True)
    bias_correction_mu: bool = field(default=True, static=True)
    bias_correction_nu: bool = field(default=True, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.scale_by_amsgrad(**as_shallow_dict(self))


@dataclass
class ScaleByBelief[Weights: PyTree](OptaxWrapper[Weights]):
    """Rescale updates according to the AdaBelief algorithm.

    References:
        Zhuang et al., 2020,
            [AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed
            Gradients](https://arxiv.org/abs/2010.07468)

    Args:
        b1: decay rate for the exponentially weighted average of grads.
        b2: decay rate for the exponentially weighted average of variance of grads.
        eps: term added to the denominator to improve numerical stability.
        eps_root: term added to the second moment of the prediction error to improve
            numerical stability.  It must be nonzero when backpropagating gradients through
            the rescaling, as in meta-learning.
        nesterov: whether to use Nesterov momentum.
    """

    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.999
    eps: RealNumeric = 1e-16
    eps_root: RealNumeric = 1e-16
    _: KW_ONLY
    nesterov: bool = field(default=False, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.scale_by_belief(**as_shallow_dict(self))


@dataclass
class ScaleByDistanceOverGradients[Weights: PyTree](OptaxWrapper[Weights]):
    """Distance-over-gradients learning rate-free optimizer.

    References:
        Ivgi et al., 2023,
            [DoG is SGD's Best Friend: A Parameter-Free Dynamic Step Size
            Schedule](https://arxiv.org/abs/2302.12022)

    Args:
        reps_rel: Used to compute initial learning rate. Recommended values are 1e-4 for models
            using batch norm, 1e-6 otherwise.
        eps: Small loading term to avoid divide-by-zero errors.
        param_dtype: Dtype for storing initial parameters.
        global_scale: Global scale factor, typically 1.0 or -1.0
    """

    reps_rel: RealNumeric = 1e-06
    eps: RealNumeric = 1e-08
    param_dtype: Any = field(default=jnp.float32, static=True)
    global_scale: RealNumeric = 1.0

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.scale_by_distance_over_gradients(**as_shallow_dict(self))


@dataclass
class ScaleByFactoredRMS[Weights: PyTree](OptaxWrapper[Weights]):
    """Scaling by a factored estimate of the gradient rms (as in Adafactor).

    References:
        Shazeer and Stern, 2018,
            [Adafactor: Adaptive Learning Rates with Sublinear Memory
            Cost](https://arxiv.org/abs/1804.04235)
        Zhai et al., 2021, [Scaling Vision Transformers](https://arxiv.org/abs/2106.04560)

    Args:
        factored: Boolean: whether to use factored second-moment estimates..
        decay_rate: Float: controls second-moment exponential decay schedule.
        step_offset: For finetuning, one may set this to the starting step-number of the fine tuning
            phase.
        min_dim_size_to_factor: Only factor accumulator if two array dimensions are at least this
            size.
        epsilon: Regularization constant for squared gradient.
        decay_rate_fn: A function that accepts the current step, the decay rate parameter and
            controls the schedule for the second momentum. Defaults to the original adafactor's
            power decay schedule. One potential shortcoming of the original schedule is the fact
            that second momentum converges to 1, which effectively freezes the second momentum. To
            prevent this the user can opt for a custom schedule that sets an upper bound for the
            second momentum, like in Zhai et al., 2021.
    """

    factored: bool = field(default=True, static=True)
    decay_rate: RealNumeric = 0.8
    step_offset: RealNumeric = 0
    min_dim_size_to_factor: int = field(default=128, static=True)
    epsilon: RealNumeric = 1e-30
    decay_rate_fn: Callable[[RealNumeric, RealNumeric], RealNumeric] = field(
        default=_decay_rate_pow, static=True
    )

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.scale_by_factored_rms(**as_shallow_dict(self))


@dataclass
class ScaleByLBFGS[Weights: PyTree](OptaxWrapper[Weights]):
    r"""Scales updates by L-BFGS.

    References:
        Nocedal and Wright, 2006 (algorithms 7.4 and 7.5),
            [Numerical Optimization](https://doi.org/10.1007/978-0-387-40065-5)
        Liu and Nocedal, 1989,
            [On the limited memory BFGS method for large scale
            optimization](https://doi.org/10.1007/BF01589116)

    Args:
        memory_size: Number of past parameters, gradients/updates to keep in memory to approximate
            the Hessian inverse.
        scale_init_precond: Whether to use a scaled identity as the initial preconditioner, see
            formula of :math:`\gamma_k` above.
    """

    memory_size: int = field(default=10, static=True)
    scale_init_precond: bool = field(default=True, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.scale_by_lbfgs(**as_shallow_dict(self))


@dataclass
class ScaleByLion[Weights: PyTree](OptaxWrapper[Weights]):
    """Rescale updates according to the Lion algorithm.

    Args:
        b1: Rate for combining the momentum and the current grad.
        b2: Decay rate for the exponentially weighted average of grads.
        mu_dtype: Optional `dtype` to be used for the momentum; if `None` then the `dtype is
            inferred from `params` and `updates`. mode : Which sign variant to use: "Hard (sign)",
            "smooth (tanh smoothing)," or "refined" (linear around 0, saturate to sign for a large
            value).
        mode: Which sign variant to use: ``"hard"`` (the sign), ``"smooth"`` (tanh
            smoothing), or ``"refined"`` (linear around zero, saturating to the sign).
        smooth_beta: Smoothing factor used when mode == "smooth"
    """

    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.99
    mu_dtype: Any = field(default=None, static=True)
    _: KW_ONLY
    mode: Literal["hard", "smooth", "refined"] = field(default="hard", static=True)
    smooth_beta: RealNumeric = 1.0

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.scale_by_lion(**as_shallow_dict(self))


@dataclass
class ScaleByNovograd[Weights: PyTree](OptaxWrapper[Weights]):
    """Computes NovoGrad updates.

    Args:
        b1: A decay rate for the exponentially weighted average of grads.
        b2: A decay rate for the exponentially weighted average of squared grads.
        eps: A term added to the denominator to improve numerical stability.
        eps_root: A term added to the denominator inside the square-root to improve numerical
            stability when backpropagating gradients through the rescaling.
        weight_decay: A scalar weight decay rate.
        mu_dtype: An optional `dtype` to be used for the first order accumulator; if `None` then the
            `dtype` is inferred from `params` and `updates`.
    """

    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.25
    eps: RealNumeric = 1e-08
    eps_root: RealNumeric = 0.0
    weight_decay: RealNumeric = 0.0
    mu_dtype: Any = field(default=None, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.scale_by_novograd(**as_shallow_dict(self))


@dataclass
class ScaleByOptimisticGradient[Weights: PyTree](OptaxWrapper[Weights]):
    """Compute generalized optimistic gradients.

    Args:
        alpha: Coefficient for generalized optimistic gradient descent.
        beta: Coefficient for negative momentum.
    """

    alpha: RealNumeric = 1.0
    beta: RealNumeric = 1.0

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.scale_by_optimistic_gradient(**as_shallow_dict(self))


@dataclass
class ScaleByParamBlockNorm[Weights: PyTree](OptaxWrapper[Weights]):
    """Scale updates for each param block by the norm of that block's parameters.

    A `block` is here a weight vector (e.g. in a Linear layer) or a weight matrix
    (e.g. in a convolutional layer) appearing as a leaf in the grads/param pytree.

    Args:
        min_scale: minimum scaling factor.
    """

    min_scale: RealNumeric = 1e-3

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.scale_by_param_block_norm(**as_shallow_dict(self))


@dataclass
class ScaleByParamBlockRMS[Weights: PyTree](OptaxWrapper[Weights]):
    """Scale updates by rms of the gradient for each param vector or matrix.

    A `block` is here a weight vector (e.g. in a Linear layer) or a weight matrix
    (e.g. in a convolutional layer) appearing as a leaf in the grads/param pytree.

    Args:
        min_scale: minimum scaling factor.
    """

    min_scale: RealNumeric = 1e-3

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.scale_by_param_block_rms(**as_shallow_dict(self))


@dataclass
class ScaleByPolyak[Weights: PyTree](OptaxWrapper[Weights]):
    r"""Scales the update by Polyak's step-size.

    Args:
        f_min: A lower bound on the objective function (defaults to 0). Corresponds to
            :math:`f^\star` in the formula above.
        max_learning_rate: A maximum step size to use (defaults to 1).
        eps: A value to add in the denominator of the update (defaults to 0).
        variant: Either ``'sps'`` or ``'sps+'`` (defaults to ``'sps'``).
    """

    f_min: RealNumeric = 0.0
    max_learning_rate: RealNumeric = 1.0
    eps: RealNumeric = 0.0
    variant: str = field(default="sps", static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.scale_by_polyak(**as_shallow_dict(self))


@dataclass
class ScaleByRAdam[Weights: PyTree](OptaxWrapper[Weights]):
    """Rescale updates according to the Rectified Adam algorithm.

    References:
        Liu et al., 2019,
            [On the Variance of the Adaptive Learning Rate and
            Beyond](https://arxiv.org/abs/1908.03265)

    Args:
        b1: decay rate for the exponentially weighted average of grads.
        b2: decay rate for the exponentially weighted average of squared grads.
        eps: term added to the denominator to improve numerical stability.
        eps_root: term added to the denominator inside the square-root to improve
            numerical stability when backpropagating gradients through the rescaling.
        threshold: Threshold for variance tractability.
        nesterov: whether to use Nesterov momentum.
    """

    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.999
    eps: RealNumeric = 1e-8
    eps_root: RealNumeric = 0.0
    threshold: RealNumeric = 5.0
    _: KW_ONLY
    nesterov: bool = field(default=False, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.scale_by_radam(**as_shallow_dict(self))


@dataclass
class ScaleByRMS[Weights: PyTree](OptaxWrapper[Weights]):
    """Rescale updates by the root of the exp. moving avg of the square.

    References:
        Tieleman and Hinton, 2012,
            [Lecture 6.5: RMSprop, Neural Networks for Machine
            Learning](https://www.cs.toronto.edu/~hinton/coursera/lecture6/lec6.pdf)

    Args:
        decay: decay rate for the exponentially weighted average of squared grads.
        eps: term added to the denominator to improve numerical stability.
        initial_scale: initial value for second moment.
        eps_in_sqrt: whether to add `eps` inside the square root of the denominator or
            outside it.
        bias_correction: whether to apply bias correction to the exponentially weighted
            average of squared grads.
    """

    decay: RealNumeric = 0.9
    eps: RealNumeric = 1e-8
    initial_scale: RealNumeric = 0.0
    eps_in_sqrt: bool = field(default=True, static=True)
    bias_correction: bool = field(default=False, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.scale_by_rms(**as_shallow_dict(self))


@dataclass
class ScaleByRprop[Weights: PyTree](OptaxWrapper[Weights]):
    """Scale with the Rprop optimizer.

    Args:
        learning_rate: The initial step size.
        eta_minus: Multiplicative factor for decreasing step size. This is applied when the gradient
            changes sign from one step to the next.
        eta_plus: Multiplicative factor for increasing step size. This is applied when the gradient
            has the same sign from one step to the next.
        min_step_size: Minimum allowed step size. Smaller steps will be clipped to this value.
        max_step_size: Maximum allowed step size. Larger steps will be clipped to this value.
    """

    learning_rate: RealNumeric
    eta_minus: RealNumeric = 0.5
    eta_plus: RealNumeric = 1.2
    min_step_size: RealNumeric = 1e-06
    max_step_size: RealNumeric = 50.0

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.scale_by_rprop(**as_shallow_dict(self))


@dataclass
class ScaleByRss[Weights: PyTree](OptaxWrapper[Weights]):
    """Rescale updates by the root of the sum of all squared gradients to date.

    References:
        Duchi et al., 2011,
            [Adaptive Subgradient Methods for Online Learning and Stochastic
            Optimization](https://jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
        McMahan and Streeter, 2010,
            [Adaptive Bound Optimization for Online Convex
            Optimization](https://arxiv.org/abs/1002.4908)

    Args:
        initial_accumulator_value: Starting value for accumulators, must be >= 0.
        eps: A small floating point value to avoid zero denominator.
    """

    initial_accumulator_value: RealNumeric = 0.1
    eps: RealNumeric = 1e-7

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.scale_by_rss(**as_shallow_dict(self))


@dataclass
class ScaleBySign[Weights: PyTree](OptaxWrapper[Weights]):
    """Compute the signs of the gradient elements."""

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.scale_by_sign(**as_shallow_dict(self))


@dataclass
class ScaleBySM3[Weights: PyTree](OptaxWrapper[Weights]):
    """Scale updates by SM3.

    References:
        Anil et al., 2019,
            [Memory-Efficient Adaptive Optimization](https://arxiv.org/abs/1901.11150)

    Args:
        b1: decay rate for the exponentially weighted average of grads.
        b2: decay rate for the exponentially weighted average of squared grads.
        eps: term added to the denominator to improve numerical stability.
    """

    b1: RealNumeric = 0.9
    b2: float = field(default=1.0, static=True)
    eps: RealNumeric = 1e-8

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.scale_by_sm3(**as_shallow_dict(self))


@dataclass
class ScaleByStddev[Weights: PyTree](OptaxWrapper[Weights]):
    """Rescale updates by the root of the centered exp. moving average of squares.

    References:
        Tieleman and Hinton, 2012,
            [Lecture 6.5: RMSprop, Neural Networks for Machine
            Learning](https://www.cs.toronto.edu/~hinton/coursera/lecture6/lec6.pdf)

    Args:
        decay: decay rate for the exponentially weighted average of squared grads.
        eps: term added to the denominator to improve numerical stability.
        initial_scale: initial value for second moment.
        eps_in_sqrt: whether to add `eps` inside the square root of the denominator or
            outside it.
        bias_correction: whether to apply bias correction to the first and second moments.
    """

    decay: RealNumeric = 0.9
    eps: RealNumeric = 1e-8
    initial_scale: RealNumeric = 0.0
    eps_in_sqrt: bool = field(default=True, static=True)
    bias_correction: bool = field(default=False, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.scale_by_stddev(**as_shallow_dict(self))


@dataclass
class ScaleByTrustRatio[Weights: PyTree](OptaxWrapper[Weights]):
    """Scale updates by the trust ratio.

    References:
        You et al., 2019,
            [Large Batch Optimization for Deep Learning: Training BERT in 76
            minutes](https://arxiv.org/abs/1904.00962)

    Args:
        min_norm: minimum norm for params and gradient norms; by default is zero.
        trust_coefficient: a multiplier for the trust ratio.
        eps: additive constant added to the denominator for numerical stability.
    """

    min_norm: RealNumeric = 0.0
    trust_coefficient: RealNumeric = 1.0
    eps: RealNumeric = 0.0

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.scale_by_trust_ratio(**as_shallow_dict(self))


@dataclass
class ScaleByYogi[Weights: PyTree](OptaxWrapper[Weights]):
    """Rescale updates according to the Yogi algorithm.

    References:
        Zaheer et al., 2018,
            [Adaptive Methods for Nonconvex
            Optimization](https://papers.nips.cc/paper/2018/hash/90365351ccc7437a1309dc64e4db32a3-Abstract.html)

    Args:
        b1: decay rate for the exponentially weighted average of grads.
        b2: decay rate for the exponentially weighted average of variance of grads.
        eps: term added to the denominator to improve numerical stability.
        eps_root: term added to the denominator inside the square-root to improve
            numerical stability when backpropagating gradients through the rescaling.
        initial_accumulator_value: The starting value for accumulators.
            Only positive values are allowed.
    """

    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.999
    eps: RealNumeric = 1e-3
    eps_root: RealNumeric = 0.0
    initial_accumulator_value: RealNumeric = 1e-6

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.scale_by_yogi(**as_shallow_dict(self))


@dataclass
class Centralize[Weights: PyTree](OptaxWrapper[Weights]):
    """Centralize gradients by subtracting their mean along the leading dimension.

    References:
        Yong et al., 2020,
            [Gradient Centralization: A New Optimization Technique for Deep Neural
            Networks](https://arxiv.org/abs/2004.01461)
    """

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.centralize(**as_shallow_dict(self))


@dataclass
class NormalizeByUpdateNorm[Weights: PyTree](OptaxWrapper[Weights]):
    """Scale by the inverse of the update norm.

    Args:
        scale_factor: Factor by which the update will be multiplied (defaults to 1).
        eps: Jitter term to avoid dividing by 0
    """

    scale_factor: RealNumeric = 1.0
    eps: RealNumeric = 1e-06

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.normalize_by_update_norm(**as_shallow_dict(self))
