"""Complete optimizers, wrapping those in ``optax._src.alias``."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import KW_ONLY
from typing import Any, override

import jax.numpy as jnp
import optax
from optax import GradientTransformation as OptaxGradientTransformation

from tjax._src.annotations import (
    IntegralNumeric,
    KeyArray,
    PyTree,
    RealNumeric,
)
from tjax.dataclasses import as_shallow_dict, dataclass, field

from .linesearch import ScaleByZoomLinesearch
from .schedule import ScalarOrSchedule
from .transform import GradientTransformation, OptaxWrapper, to_optax_or_none


@dataclass
class AdaBelief[Weights: PyTree](OptaxWrapper[Weights]):
    """Pytree-compatible wrapper for the optax AdaBelief optimiser.

    References:
        Zhuang et al., 2020,
            [AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed
            Gradients](https://arxiv.org/abs/2010.07468)

    Args:
        learning_rate: Global step-size schedule or scalar.
        b1: Decay rate for the first moment (mean) of gradients.
        b2: Decay rate for the second moment (variance) of gradients.
        eps: Regularisation term for numerical stability.
        eps_root: Term added to the second moment of the prediction error for numerical
            stability.  It must be nonzero when backpropagating through the optimiser.
        nesterov: Whether to use Nesterov momentum.
    """

    learning_rate: ScalarOrSchedule
    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.999
    eps: RealNumeric = 1e-16
    eps_root: RealNumeric = 1e-16
    _: KW_ONLY
    nesterov: bool = field(default=False, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.adabelief(**as_shallow_dict(self))


@dataclass
class AdaDelta[Weights: PyTree](OptaxWrapper[Weights]):
    """Pytree-compatible wrapper for the optax AdaDelta optimiser.

    References:
        Zeiler, 2012, [ADADELTA: An Adaptive Learning Rate Method](https://arxiv.org/abs/1212.5701)

    Args:
        learning_rate: Optional global scaling factor.
        rho: Decay rate for the running average of squared gradients.
        eps: Regularisation term for numerical stability.
        weight_decay: Coefficient for L2 regularisation.
        weight_decay_mask: Optional mask controlling which parameters
            receive weight decay.
    """

    learning_rate: ScalarOrSchedule | None = None
    rho: RealNumeric = 0.9
    eps: RealNumeric = 1e-6
    weight_decay: ScalarOrSchedule = 0.0
    weight_decay_mask: bool | Weights | Callable[[Weights], Any] | None = field(
        default=None, static=True
    )

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.adadelta(**as_shallow_dict(self))


@dataclass
class AdaFactor[Weights: PyTree](OptaxWrapper[Weights]):
    """Pytree-compatible wrapper for the optax AdaFactor optimiser.

    AdaFactor reduces memory usage by factoring the second-moment estimate.

    References:
        Shazeer and Stern, 2018,
            [Adafactor: Adaptive Learning Rates with Sublinear Memory
            Cost](https://arxiv.org/abs/1804.04235)

    Args:
        learning_rate: Optional global step-size schedule or scalar.  AdaFactor's natural
            learning-rate scale differs markedly from Adam's.
        min_dim_size_to_factor: Factor the statistics only if two array dimensions have at
            least this size.
        decay_rate: Controls the second-moment exponential decay schedule.
        decay_offset: For fine-tuning, the step number at which fine-tuning starts.
        multiply_by_parameter_scale: Whether to scale the learning rate by the parameter norm;
            if false, the learning rate is the absolute step size.
        clipping_threshold: Optional clipping threshold, at least 1; ``None`` disables
            clipping.
        momentum: Optional momentum decay rate between 0 and 1; ``None`` disables momentum,
            which otherwise uses extra memory.
        dtype_momentum: Data type of the momentum buffers.
        weight_decay_rate: Optional weight decay rate, or a schedule of it.
        eps: Regularisation constant for the root mean squared gradient.
        factored: Whether to use factored second-moment estimates.
        weight_decay_mask: Optional mask, or a function of the parameters returning one,
            selecting which parameters receive weight decay.
    """

    learning_rate: ScalarOrSchedule | None = None
    min_dim_size_to_factor: int = field(default=128, static=True)
    decay_rate: RealNumeric = 0.8
    decay_offset: IntegralNumeric = 0
    multiply_by_parameter_scale: bool = field(default=True, static=True)
    clipping_threshold: float | None = 1.0
    momentum: float | None = None
    dtype_momentum: Any = field(default=jnp.float32, static=True)
    weight_decay_rate: ScalarOrSchedule | None = None
    eps: RealNumeric = 1e-30
    factored: bool = field(default=True, static=True)
    weight_decay_mask: bool | Weights | Callable[[Weights], Any] | None = field(
        default=None, static=True
    )

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.adafactor(**as_shallow_dict(self))


@dataclass
class AdaGrad[Weights: PyTree](OptaxWrapper[Weights]):
    """Pytree-compatible wrapper for the optax AdaGrad optimiser.

    References:
        Duchi et al., 2011,
            [Adaptive Subgradient Methods for Online Learning and Stochastic
            Optimization](https://jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)

    Args:
        learning_rate: Global step-size schedule or scalar.
        initial_accumulator_value: Starting value for the sum-of-squares accumulator.
        eps: Small constant added inside the square root of the denominator.
    """

    learning_rate: ScalarOrSchedule
    initial_accumulator_value: RealNumeric = 0.1
    eps: RealNumeric = 1e-7

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.adagrad(**as_shallow_dict(self))


@dataclass
class Adam[Weights: PyTree](OptaxWrapper[Weights]):
    """Pytree-compatible wrapper for the optax Adam optimiser.

    References:
        Kingma and Ba, 2014,
            [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)

    Args:
        learning_rate: Global step-size schedule or scalar.
        b1: Decay rate for the first moment of gradients.
        b2: Decay rate for the second moment of gradients.
        eps: Regularisation term for numerical stability.
        eps_root: Term added inside the square-root for numerical stability.
        mu_dtype: Optional dtype for the first moment accumulator.
        nesterov: Whether to use Nesterov momentum.
    """

    learning_rate: ScalarOrSchedule
    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.999
    eps: RealNumeric = 1e-8
    eps_root: RealNumeric = 0.0
    mu_dtype: Any | None = field(default=None, static=True)
    _: KW_ONLY
    nesterov: bool = field(default=False, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.adam(**as_shallow_dict(self))


@dataclass
class AdamW[Weights: PyTree](OptaxWrapper[Weights]):
    """Pytree-compatible wrapper for the optax AdamW optimiser.

    AdamW decouples the weight decay from the gradient-based update.

    References:
        Loshchilov and Hutter, 2017,
            [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)

    Args:
        learning_rate: Global step-size schedule or scalar.
        b1: Decay rate for the first moment of gradients.
        b2: Decay rate for the second moment of gradients.
        eps: Regularisation term for numerical stability.
        eps_root: Term added inside the square-root for numerical stability.
        mu_dtype: Optional dtype for the first moment accumulator.
        weight_decay: Strength of the decoupled weight decay, or a schedule of it.  As in
            PyTorch, it is multiplied by the learning rate.
        mask: Optional mask, or a function of the parameters returning one, selecting
            which parameters receive weight decay.
        nesterov: Whether to use Nesterov momentum.
    """

    learning_rate: ScalarOrSchedule
    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.999
    eps: RealNumeric = 1e-8
    eps_root: RealNumeric = 0.0
    mu_dtype: Any | None = field(default=None, static=True)
    weight_decay: ScalarOrSchedule = 1e-4
    mask: bool | Weights | Callable[[Weights], Any] | None = field(default=None, static=True)
    _: KW_ONLY
    nesterov: bool = field(default=False, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.adamw(**as_shallow_dict(self))


@dataclass
class Adamax[Weights: PyTree](OptaxWrapper[Weights]):
    """Pytree-compatible wrapper for the optax Adamax optimiser.

    Adamax is a variant of Adam based on the infinity norm.

    References:
        Kingma and Ba, 2014,
            [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)

    Args:
        learning_rate: Global step-size schedule or scalar.
        b1: Decay rate for the first moment of gradients.
        b2: Decay rate for the infinity norm of gradients.
        eps: Regularisation term for numerical stability.
    """

    learning_rate: ScalarOrSchedule
    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.999
    eps: RealNumeric = 1e-8

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.adamax(**as_shallow_dict(self))


@dataclass
class AdamaxW[Weights: PyTree](OptaxWrapper[Weights]):
    """Pytree-compatible wrapper for the optax AdamaxW optimiser.

    AdamaxW combines the infinity-norm update of Adamax with decoupled weight
    decay from AdamW.

    Args:
        learning_rate: Global step-size schedule or scalar.
        b1: Decay rate for the first moment of gradients.
        b2: Decay rate for the infinity norm of gradients.
        eps: Regularisation term for numerical stability.
        weight_decay: Strength of the decoupled weight decay, or a schedule of it.  As in
            PyTorch, it is multiplied by the learning rate.
        mask: Optional mask, or a function of the parameters returning one, selecting
            which parameters receive weight decay.
    """

    learning_rate: ScalarOrSchedule
    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.999
    eps: RealNumeric = 1e-8
    weight_decay: ScalarOrSchedule = 1e-4
    mask: bool | Weights | Callable[[Weights], Any] | None = field(default=None, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.adamaxw(**as_shallow_dict(self))


@dataclass
class Adan[Weights: PyTree](OptaxWrapper[Weights]):
    """The ADAptive Nesterov momentum algorithm (Adan).

    References:
        Xie et al., 2022,
            [Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep
            Models](https://arxiv.org/abs/2208.06677)

    Args:
        learning_rate: This is a fixed global scaling factor.
        b1: Decay rate for the EWMA of gradients.
        b2: Decay rate for the EWMA of differences of gradients.
        b3: Decay rate for the EMWA of the algorithm's squared term.
        eps: Term added to the denominator to improve numerical stability.
        eps_root: Term added to the denominator inside the square-root to improve numerical
            stability when backpropagating gradients through the rescaling.
        weight_decay: Strength of the weight decay regularization.
        mask: A tree with same structure as (or a prefix of) the params PyTree, or a Callable that
            returns such a pytree given the params/updates. The leaves should be booleans, `True`
            for leaves/subtrees you want to apply the weight decay to, and `False` for those you
            want to skip.
    """

    learning_rate: ScalarOrSchedule
    b1: RealNumeric = 0.98
    b2: RealNumeric = 0.92
    b3: RealNumeric = 0.99
    eps: RealNumeric = 1e-08
    eps_root: RealNumeric = 1e-08
    weight_decay: ScalarOrSchedule = 0.0
    mask: bool | Weights | Callable[[Weights], Any] | None = field(default=None, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.adan(**as_shallow_dict(self))


@dataclass
class AMSGrad[Weights: PyTree](OptaxWrapper[Weights]):
    """The AMSGrad optimizer.

    References:
        Reddi et al., 2018,
            [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ)

    Args:
        learning_rate: A global scaling factor, either fixed or evolving along iterations with a
            scheduler, see :func:`optax.scale_by_learning_rate`.
        b1: Exponential decay rate to track the first moment of past gradients.
        b2: Exponential decay rate to track the second moment of past gradients.
        eps: A small constant applied to denominator outside of the square root (as in the Adam
            paper) to avoid dividing by zero when rescaling.
        eps_root: A small constant applied to denominator inside the square root (as in RMSProp), to
            avoid dividing by zero when rescaling. This is needed for instance when computing
            (meta-)gradients through Adam.
        mu_dtype: Optional `dtype` to be used for the first order accumulator; if `None` then the
            `dtype` is inferred from `params` and `updates`.
        bias_correction_mu: Whether to apply bias correction to the first moment estimate. Set to
            ``False`` to match the original AMSGrad paper.
        bias_correction_nu: Whether to apply bias correction to the second moment estimate before
            taking the elementwise maximum (``nu_max``). Set to ``False`` to match the original
            AMSGrad paper.
    """

    learning_rate: ScalarOrSchedule
    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.999
    eps: RealNumeric = 1e-08
    eps_root: RealNumeric = 0.0
    mu_dtype: Any = field(default=None, static=True)
    bias_correction_mu: bool = field(default=True, static=True)
    bias_correction_nu: bool = field(default=True, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.amsgrad(**as_shallow_dict(self))


@dataclass
class Fromage[Weights: PyTree](OptaxWrapper[Weights]):
    """Pytree-compatible wrapper for the optax Fromage optimiser.

    Fromage keeps parameter updates within a trust region defined by the
    relative distance in function space.

    References:
        Bernstein et al., 2020,
            [On the distance between two neural networks and the stability of
            learning](https://arxiv.org/abs/2002.03432)

    Args:
        learning_rate: Global step-size schedule or scalar.
        min_norm: Lower bound on the norms of the updates and parameters, which avoids
            dividing by zero in the trust ratio.
    """

    learning_rate: ScalarOrSchedule
    min_norm: RealNumeric = 1e-6

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.fromage(**as_shallow_dict(self))


@dataclass
class LAMB[Weights: PyTree](OptaxWrapper[Weights]):
    """Pytree-compatible wrapper for the optax LAMB optimiser.

    LAMB scales Adam updates by the ratio of the parameter norm to the update
    norm, which enables large-batch training.

    References:
        You et al., 2019,
            [Large Batch Optimization for Deep Learning: Training BERT in 76
            minutes](https://arxiv.org/abs/1904.00962)

    Args:
        learning_rate: Global step-size schedule or scalar.
        b1: Decay rate for the first moment of gradients.
        b2: Decay rate for the second moment of gradients.
        eps: Regularisation term for numerical stability.
        eps_root: Term added inside the square-root for numerical stability.
        weight_decay: Strength of the decoupled weight decay, applied after the Adam
            scaling and before the trust ratio.
        mask: Optional mask, or a function of the parameters returning one, selecting
            which parameters receive weight decay.
    """

    learning_rate: ScalarOrSchedule
    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.999
    eps: RealNumeric = 1e-6
    eps_root: RealNumeric = 0.0
    weight_decay: ScalarOrSchedule = 0.0
    mask: bool | Weights | Callable[[Weights], Any] | None = field(default=None, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.lamb(**as_shallow_dict(self))


@dataclass
class LARS[Weights: PyTree](OptaxWrapper[Weights]):
    """Pytree-compatible wrapper for the optax LARS optimiser.

    Layer-wise Adaptive Rate Scaling adapts the learning rate per layer based
    on the ratio of the parameter norm to the gradient norm.

    References:
        You et al., 2017,
            [Large Batch Training of Convolutional Networks](https://arxiv.org/abs/1708.03888)

    Args:
        learning_rate: Global step-size schedule or scalar.
        weight_decay: Coefficient for L2 regularisation.
        weight_decay_mask: Mask selecting parameters that receive weight decay.
        trust_coefficient: Scaling factor for the trust ratio.
        eps: Regularisation term added to the gradient norm.
        trust_ratio_mask: Mask selecting parameters that use the trust ratio.
        momentum: Momentum decay rate.
        nesterov: Whether to use Nesterov momentum.
    """

    learning_rate: ScalarOrSchedule
    weight_decay: ScalarOrSchedule = 0.0
    weight_decay_mask: bool | Weights | Callable[[Weights], Any] | None = field(
        default=True, static=True
    )
    trust_coefficient: RealNumeric = 0.001
    eps: RealNumeric = 0.0
    trust_ratio_mask: bool | Weights | Callable[[Weights], Any] | None = field(
        default=True, static=True
    )
    momentum: RealNumeric = 0.9
    nesterov: bool = field(default=False, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.lars(**as_shallow_dict(self))


@dataclass
class LBFGS[Weights: PyTree](OptaxWrapper[Weights]):
    """Pytree-compatible wrapper for the optax L-BFGS optimiser.

    Limited-memory BFGS approximates the inverse Hessian using a fixed-size
    history of gradient and parameter differences.  With a linesearch, pass the loss, its
    gradient and the loss function to ``update`` as ``value``, ``grad`` and ``value_fn``.

    Args:
        learning_rate: Optional step-size schedule or scalar.  When ``None``
            the linesearch determines the step size.
        memory_size: Number of past gradient/step pairs to retain.
        scale_init_precond: Whether to scale the initial preconditioner.
        linesearch: Optional linesearch transformation; defaults to
            ``ScaleByZoomLinesearch`` with 15 maximum steps.
    """

    learning_rate: ScalarOrSchedule | None = None
    memory_size: int = field(default=10, static=True)
    scale_init_precond: bool = field(default=True, static=True)
    linesearch: GradientTransformation[Any, Weights] | None = field(
        default_factory=lambda: ScaleByZoomLinesearch(max_linesearch_steps=15)
    )

    @override
    def _optax(self) -> OptaxGradientTransformation:
        kwargs = as_shallow_dict(self)
        kwargs["linesearch"] = to_optax_or_none(self.linesearch)
        return optax.lbfgs(**kwargs)


@dataclass
class Lion[Weights: PyTree](OptaxWrapper[Weights]):
    """Pytree-compatible wrapper for the optax Lion optimiser.

    Lion (Evolved Sign Momentum) uses the sign of the update rather than the
    magnitude, reducing memory by eliminating the second-moment accumulator.

    References:
        Chen et al., 2023,
            [Symbolic Discovery of Optimization Algorithms](https://arxiv.org/abs/2302.06675)

    Args:
        learning_rate: Global step-size schedule or scalar.
        b1: Rate at which to combine the momentum and the current gradient before taking
            the sign.
        b2: Decay rate for the momentum of past gradients.
        mu_dtype: Optional dtype for the momentum accumulator.
        weight_decay: Strength of the decoupled weight decay, or a schedule of it.  As in
            PyTorch, it is multiplied by the learning rate.
        mask: Optional mask, or a function of the parameters returning one, selecting
            which parameters receive weight decay.
    """

    learning_rate: ScalarOrSchedule
    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.99
    mu_dtype: Any | None = field(default=None, static=True)
    weight_decay: ScalarOrSchedule = 1e-3
    mask: bool | Weights | Callable[[Weights], Any] | None = field(default=None, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.lion(**as_shallow_dict(self))


@dataclass
class NAdam[Weights: PyTree](OptaxWrapper[Weights]):
    """The NAdam optimizer.

    References:
        Dozat, 2016,
            [Incorporating Nesterov Momentum into
            Adam](https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ)

    Args:
        learning_rate: A global scaling factor, either fixed or evolving along iterations with a
            scheduler, see :func:`optax.scale_by_learning_rate`.
        b1: Exponential decay rate to track the first moment of past gradients.
        b2: Exponential decay rate to track the second moment of past gradients.
        eps: A small constant applied to denominator outside of the square root (as in the Adam
            paper) to avoid dividing by zero when rescaling.
        eps_root: A small constant applied to denominator inside the square root (as in RMSProp), to
            avoid dividing by zero when rescaling. This is needed for example when computing
            (meta-)gradients through Adam.
        mu_dtype: Optional `dtype` to be used for the first order accumulator; if `None` then the
            `dtype` is inferred from `params` and `updates`.
        nesterov: Whether to use Nesterov momentum.
    """

    learning_rate: ScalarOrSchedule
    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.999
    eps: RealNumeric = 1e-08
    eps_root: RealNumeric = 0.0
    mu_dtype: Any = field(default=None, static=True)
    _: KW_ONLY
    nesterov: bool = field(default=True, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.nadam(**as_shallow_dict(self))


@dataclass
class NAdamW[Weights: PyTree](OptaxWrapper[Weights]):
    """NAdamW optimizer, implemented as part of the AdamW optimizer.

    References:
        Loshchilov and Hutter, 2017,
            [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
        Dozat, 2016,
            [Incorporating Nesterov Momentum into
            Adam](https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ)

    Args:
        learning_rate: A global scaling factor, either fixed or evolving along iterations with a
            scheduler, see :func:`optax.scale_by_learning_rate`.
        b1: Exponential decay rate to track the first moment of past gradients.
        b2: Exponential decay rate to track the second moment of past gradients.
        eps: A small constant applied to denominator outside of the square root (as in the Adam
            paper) to avoid dividing by zero when rescaling.
        eps_root: A small constant applied to denominator inside the square root (as in RMSProp), to
            avoid dividing by zero when rescaling. This is needed for instance when computing
            (meta-)gradients through Adam.
        mu_dtype: Optional `dtype` to be used for the first order accumulator; if `None` then the
            `dtype` is inferred from `params` and `updates`.
        weight_decay: Strength of the weight decay regularization. Note that this weight decay is
            multiplied with the learning rate. This is consistent with other frameworks such as
            PyTorch, but different from (Loshchilov et al, 2019) where the weight decay is only
            multiplied with the "schedule multiplier", but not the base learning rate.
        mask: A tree with same structure as (or a prefix of) the params PyTree, or a Callable that
            returns such a pytree given the params/updates. The leaves should be booleans, `True`
            for leaves/subtrees you want to apply the weight decay to, and `False` for those you
            want to skip. Note that the Adam gradient transformations are applied to all parameters.
        nesterov: Whether to use Nesterov momentum.
    """

    learning_rate: ScalarOrSchedule
    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.999
    eps: RealNumeric = 1e-08
    eps_root: RealNumeric = 0.0
    mu_dtype: Any = field(default=None, static=True)
    weight_decay: ScalarOrSchedule = 0.0001
    mask: bool | Weights | Callable[[Weights], Any] | None = field(default=None, static=True)
    _: KW_ONLY
    nesterov: bool = field(default=True, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.nadamw(**as_shallow_dict(self))


@dataclass
class NoisySGD[Weights: PyTree](OptaxWrapper[Weights]):
    """Pytree-compatible wrapper for the optax Noisy SGD optimiser.

    Adds annealed Gaussian noise to the gradient at each step.

    References:
        Neelakantan et al., 2015,
            [Adding Gradient Noise Improves Learning for Very Deep
            Networks](https://arxiv.org/abs/1511.06807)

    Args:
        learning_rate: Global step-size schedule or scalar.
        eta: Base variance of the Gaussian noise.
        gamma: Decay exponent for the noise variance, which decays as
            ``(1 + t) ** -gamma`` at step ``t``.
        key: PRNG key or integer seed for random noise generation.
    """

    learning_rate: ScalarOrSchedule
    eta: RealNumeric = 0.01
    gamma: RealNumeric = 0.55
    key: KeyArray | int | None = None

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.noisy_sgd(**as_shallow_dict(self))


@dataclass
class Novograd[Weights: PyTree](OptaxWrapper[Weights]):
    """Pytree-compatible wrapper for the optax NovoGrad optimiser.

    NovoGrad normalises gradients layer-wise using the second moment,
    and uses a first moment similar to Adam.

    References:
        Ginsburg et al., 2019,
            [Stochastic Gradient Methods with Layer-wise Adaptive Moments for Training of Deep
            Networks](https://arxiv.org/abs/1905.11286)

    Args:
        learning_rate: Global step-size schedule or scalar.
        b1: Decay rate for the first moment of gradients.
        b2: Decay rate for the second moment of gradients.
        eps: Regularisation term for numerical stability.
        eps_root: Term added inside the square-root for numerical stability.
        weight_decay: Strength of the weight decay.
    """

    learning_rate: ScalarOrSchedule
    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.25
    eps: RealNumeric = 1e-6
    eps_root: RealNumeric = 0.0
    # optax crashes on a schedule here: https://github.com/google-deepmind/optax/issues/1857
    weight_decay: ScalarOrSchedule = 0.0

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.novograd(**as_shallow_dict(self))


@dataclass
class OptimisticAdam[Weights: PyTree](OptaxWrapper[Weights]):
    """The Optimistic Adam optimizer.

    References:
        Daskalakis et al., 2017, [Training GANs with Optimism](https://arxiv.org/abs/1711.00141)

    Args:
        learning_rate: A global scaling factor, either fixed or evolving along iterations with a
            scheduler, see :func:`optax.scale_by_learning_rate`.
        alpha: One of two scalar optimism parameters in optimistic gradient descent.
        beta: One of two scalar optimism parameters in optimistic gradient descent.
        b1: Exponential decay rate to track the first moment of past gradients.
        b2: Exponential decay rate to track the second moment of past gradients.
        eps: Term added to the denominator to improve numerical stability.
        eps_root: Term added to the second moment of the prediction error to improve numerical
            stability. If backpropagating gradients through the gradient transformation (e.g. for
            meta-learning), this must be non-zero.
        mu_dtype: Optional `dtype` to be used for the first order accumulator; if `None` then the
            `dtype` is inferred from `params` and `updates`.
        nesterov: Whether to use Nesterov momentum.
    """

    learning_rate: ScalarOrSchedule
    _: KW_ONLY
    alpha: RealNumeric = 1.0
    beta: RealNumeric = 1.0
    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.999
    eps: RealNumeric = 1e-08
    eps_root: RealNumeric = 0.0
    mu_dtype: Any = field(default=None, static=True)
    nesterov: bool = field(default=True, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.optimistic_adam_v2(**as_shallow_dict(self))


@dataclass
class OptimisticGradientDescent[Weights: PyTree](OptaxWrapper[Weights]):
    """Pytree-compatible wrapper for the optax Optimistic Gradient Descent optimiser.

    Optimistic gradient descent extrapolates from the previous gradient, which stabilises
    training in min-max games.  Before the learning rate, the update is
    ``(alpha + beta) * g_t - beta * g_{t-1}``.

    References:
        Daskalakis et al., 2017, [Training GANs with Optimism](https://arxiv.org/abs/1711.00141)

    Args:
        learning_rate: Global step-size schedule or scalar.
        alpha: Coefficient of the current gradient.
        beta: Coefficient of the negative momentum, which subtracts the previous gradient.
    """

    learning_rate: ScalarOrSchedule
    # optax crashes on a schedule for these: https://github.com/google-deepmind/optax/issues/1857
    alpha: ScalarOrSchedule = 1.0
    beta: ScalarOrSchedule = 1.0

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.optimistic_gradient_descent(**as_shallow_dict(self))


@dataclass
class PolyakSGD[Weights: PyTree](OptaxWrapper[Weights]):
    """Pytree-compatible wrapper for the optax Polyak SGD optimiser.

    The step size is ``scaling * (f(x) - f_min) / (||grad f(x)||^2 + eps)``, capped at
    ``max_learning_rate``.  Pass the loss at the current parameters to ``update`` as ``value``.

    References:
        Polyak, 1987, Introduction to Optimization
        Loizou et al., 2020,
            [Stochastic Polyak Step-size for SGD: An Adaptive Learning Rate for Fast
            Convergence](https://arxiv.org/abs/2002.10542)
        Garrigos et al., 2023,
            [Function Value Learning: Adaptive Learning Rates Based on the Polyak Stepsize and
            Function Splitting in ERM](https://arxiv.org/abs/2307.14528)

    Args:
        max_learning_rate: Upper bound on the computed step size.
        scaling: Optional schedule or scalar to scale the Polyak step.
        f_min: Known or estimated minimum value of the loss.
        eps: Regularisation term added to the gradient norm denominator.
        variant: Either ``'sps'``, or ``'sps+'``, which uses only the nonnegative part of
            ``f(x) - f_min``.
    """

    max_learning_rate: RealNumeric = 1.0
    scaling: ScalarOrSchedule = 1.0
    f_min: RealNumeric = 0.0
    eps: RealNumeric = 0.0
    variant: str = field(default="sps", static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.polyak_sgd(**as_shallow_dict(self))


@dataclass
class RAdam[Weights: PyTree](OptaxWrapper[Weights]):
    """Pytree-compatible wrapper for the optax RAdam optimiser.

    Rectified Adam stabilises Adam's warm-up phase by rectifying the variance
    of the adaptive learning rate.

    References:
        Liu et al., 2019,
            [On the Variance of the Adaptive Learning Rate and
            Beyond](https://arxiv.org/abs/1908.03265)

    Args:
        learning_rate: Global step-size schedule or scalar.
        b1: Decay rate for the first moment of gradients.
        b2: Decay rate for the second moment of gradients.
        eps: Regularisation term for numerical stability.
        eps_root: Term added inside the square-root for numerical stability.
        threshold: Minimum variance tractability threshold.
        nesterov: Whether to use Nesterov momentum.
    """

    learning_rate: ScalarOrSchedule
    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.999
    eps: RealNumeric = 1e-8
    eps_root: RealNumeric = 0.0
    threshold: RealNumeric = 5.0
    _: KW_ONLY
    nesterov: bool = field(default=False, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.radam(**as_shallow_dict(self))


@dataclass
class RMSProp[Weights: PyTree](OptaxWrapper[Weights]):
    """A flexible RMSProp optimiser.

    RMSProp is an SGD variant with learning rate adaptation. The `learning_rate`
    used for each weight is scaled by a suitable estimate of the magnitude of the
    gradients on previous steps. Several variants of RMSProp can be found
    in the literature. This alias provides an easy to configure RMSProp
    optimiser that can be used to switch between several of these variants.

    References:
        Tieleman and Hinton, 2012,
            [Lecture 6.5: RMSprop, Neural Networks for Machine
            Learning](https://www.cs.toronto.edu/~hinton/coursera/lecture6/lec6.pdf)
        Graves, 2013,
            [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/abs/1308.0850)

    Args:
        learning_rate: Global step-size schedule or scalar.
        decay: The decay used to track the magnitude of previous gradients.
        eps: A small numerical constant to avoid dividing by zero when rescaling.
        initial_scale: Initial value of the accumulators tracking the magnitude of previous
            updates.  PyTorch uses 0 and TF1 uses 1; when reproducing results from a paper,
            verify the value the authors used.
        eps_in_sqrt: Whether to add ``eps`` inside the square root of the denominator or
            outside it.
        centered: Whether to rescale by the variance of past gradients instead of their
            second moment.
        momentum: The decay rate of the momentum term; ``None`` disables momentum.
        nesterov: Whether to use Nesterov momentum.
        bias_correction: Whether to apply bias correction to the second-moment estimate, and
            to the first-moment estimate when ``centered`` is true.
    """

    learning_rate: ScalarOrSchedule
    decay: RealNumeric = 0.9
    eps: RealNumeric = 1e-8
    initial_scale: RealNumeric = 0.0
    eps_in_sqrt: bool = field(default=True, static=True)
    centered: bool = field(default=False, static=True)
    momentum: float | None = None
    nesterov: bool = field(default=False, static=True)
    bias_correction: bool = field(default=False, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.rmsprop(**as_shallow_dict(self))


@dataclass
class Rprop[Weights: PyTree](OptaxWrapper[Weights]):
    """The Rprop optimizer.

    References:
        Riedmiller and Braun, 1993,
            [A direct adaptive method for faster backpropagation learning: the RPROP
            algorithm](https://doi.org/10.1109/ICNN.1993.298623)
        Igel and Hüsken, 2003,
            [Empirical evaluation of the improved Rprop learning
            algorithms](https://doi.org/10.1016/S0925-2312(01)00700-7)

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
        return optax.rprop(**as_shallow_dict(self))


@dataclass
class SGD[Weights: PyTree](OptaxWrapper[Weights]):
    """A canonical Stochastic Gradient Descent optimiser.

    This implements stochastic gradient descent. It also includes support for
    momentum, and nesterov acceleration, as these are standard practice when
    using stochastic gradient descent to train deep neural networks.

    References:
        Sutskever et al., 2013,
            [On the importance of initialization and momentum in deep
            learning](https://proceedings.mlr.press/v28/sutskever13.pdf)

    Args:
        learning_rate: Global step-size schedule or scalar.
        momentum: The decay rate of the momentum term; ``None`` disables momentum.
        nesterov: Whether to use Nesterov momentum.
        accumulator_dtype: Optional dtype for the accumulator; if ``None``, the dtype is
            inferred from the parameters and updates.
    """

    learning_rate: ScalarOrSchedule
    momentum: float | None = None
    nesterov: bool = field(default=False, static=True)
    accumulator_dtype: Any | None = field(default=None, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.sgd(**as_shallow_dict(self))


@dataclass
class SignSGD[Weights: PyTree](OptaxWrapper[Weights]):
    """A variant of SGD using only the signs of the gradient components.

    References:
        Bernstein et al., 2018,
            [signSGD: Compressed Optimisation for Non-Convex
            Problems](https://arxiv.org/abs/1802.04434)
        Balles et al., 2020,
            [The Geometry of Sign Gradient Descent](https://arxiv.org/abs/2002.08056)

    Args:
        learning_rate: A global scaling factor, either fixed or evolving along iterations with a
            scheduler, see :func:`optax.scale_by_learning_rate`.
    """

    learning_rate: ScalarOrSchedule

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.sign_sgd(**as_shallow_dict(self))


@dataclass
class Signum[Weights: PyTree](OptaxWrapper[Weights]):
    """A variant of SGD using signs of the components of an EMA of the gradient.

    References:
        Bernstein et al., 2018,
            [signSGD: Compressed Optimisation for Non-Convex
            Problems](https://arxiv.org/abs/1802.04434)
        Zhao et al., 2024,
            [Deconstructing What Makes a Good Optimizer for Language
            Models](https://arxiv.org/abs/2407.07972)

    Args:
        learning_rate: A global scaling factor, either fixed or evolving along iterations with a
            scheduler, see :func:`optax.scale_by_learning_rate`.
        beta: Exponential moving average decay rate.
        accumulator_dtype: Data type for the EMA accumulator.
    """

    learning_rate: ScalarOrSchedule
    beta: RealNumeric = 0.9
    accumulator_dtype: Any = field(default=None, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.signum(**as_shallow_dict(self))


@dataclass
class SM3[Weights: PyTree](OptaxWrapper[Weights]):
    """Pytree-compatible wrapper for the optax SM3 optimiser.

    SM3 reduces memory by sharing accumulators across dimensions, making it
    suitable for very large parameter matrices.

    References:
        Anil et al., 2019,
            [Memory-Efficient Adaptive Optimization](https://arxiv.org/abs/1901.11150)

    Args:
        learning_rate: Global step-size scalar.
        momentum: Momentum decay rate.
    """

    learning_rate: RealNumeric
    momentum: RealNumeric = 0.9

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.sm3(**as_shallow_dict(self))


@dataclass
class Yogi[Weights: PyTree](OptaxWrapper[Weights]):
    """Pytree-compatible wrapper for the optax Yogi optimiser.

    Yogi replaces Adam's additive second-moment update with a sign-based
    update to prevent the accumulator from growing too fast.

    References:
        Zaheer et al., 2018,
            [Adaptive Methods for Nonconvex
            Optimization](https://papers.nips.cc/paper/2018/hash/90365351ccc7437a1309dc64e4db32a3-Abstract.html)

    Args:
        learning_rate: Global step-size schedule or scalar.
        b1: Decay rate for the first moment of gradients.
        b2: Decay rate for the second moment of gradients.
        eps: Regularisation term for numerical stability.
    """

    learning_rate: ScalarOrSchedule
    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.999
    eps: RealNumeric = 1e-3

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.yogi(**as_shallow_dict(self))
