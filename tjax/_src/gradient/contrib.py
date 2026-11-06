"""Optimizers and transformations wrapping those in ``optax.contrib``."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import KW_ONLY
from typing import Any, Literal, override

import optax.contrib
from optax import (
    GradientTransformation as OptaxGradientTransformation,
)

from tjax._src.annotations import (
    IntegralNumeric,
    JaxArray,
    KeyArray,
    PyTree,
    RealNumeric,
)
from tjax.dataclasses import as_shallow_dict, dataclass, field

from .schedule import ScalarOrSchedule
from .transform import (
    GradientTransformation,
    OptaxWrapper,
    to_optax_or_none,
)


def _adopt_clip_value(step: JaxArray) -> JaxArray:
    """Clipping threshold for ADOPT, as in optax's default."""
    return step**0.25


@dataclass
class ACProp[Weights: PyTree](OptaxWrapper[Weights]):
    """The ACProp optimizer.

    References:
        Zhuang et al., 2021,
            [Momentum Centering and Asynchronous Update for Adaptive Gradient
            Methods](https://arxiv.org/abs/2110.05454)

    Args:
        learning_rate: A global scaling factor, either fixed or evolving along iterations with a
            scheduler, see :func:`optax.scale_by_learning_rate`.
        b1: Exponential decay rate to track the first moment of past gradients.
        b2: Exponential decay rate to track the second moment of past gradients.
        eps: Term added to the denominator to improve numerical stability.
        eps_root: Term added to the second moment of the prediction error to improve numerical
            stability. If backpropagating gradients through the gradient transformation (e.g. for
            meta-learning), this must be non-zero.
        weight_decay: Strength of the weight decay regularization. Note that this weight decay is
            multiplied with the learning rate. This is consistent with other frameworks such as
            PyTorch, but different from (Loshchilov et al, 2019) where the weight decay is only
            multiplied with the "schedule multiplier", but not the base learning rate.
        mask: A tree with same structure as (or a prefix of) the params PyTree, or a Callable that
            returns such a pytree given the params/updates. The leaves should be booleans, `True`
            for leaves/subtrees you want to apply the weight decay to, and `False` for those you
            want to skip. Note that the Adam gradient transformations are applied to all parameters.
    """

    learning_rate: ScalarOrSchedule
    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.999
    eps: RealNumeric = 1e-16
    eps_root: RealNumeric = 1e-16
    weight_decay: RealNumeric = 0.0
    mask: bool | Weights | Callable[[Weights], Any] | None = field(default=None, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.contrib.acprop(**as_shallow_dict(self))


@dataclass
class ScaleByACProp[Weights: PyTree](OptaxWrapper[Weights]):
    """Rescale updates according to ACProp (asynchronous version of AdaBelief).

    Args:
        b1: Decay rate for the exponentially weighted average of grads.
        b2: Decay rate for the exponentially weighted average of variance of grads.
        eps: Term added to the denominator to improve numerical stability.
        eps_root: Term added to the second moment of the prediction error to improve numerical
            stability. If backpropagating gradients through the gradient transformation (e.g. for
            meta-learning), this must be non-zero.
    """

    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.999
    eps: RealNumeric = 1e-16
    eps_root: RealNumeric = 1e-16

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.contrib.scale_by_acprop(**as_shallow_dict(self))


@dataclass
class AdEMAMix[Weights: PyTree](OptaxWrapper[Weights]):
    """AdEMAMix.

    References:
        Pagliardini et al., 2024,
            [The AdEMAMix Optimizer: Better, Faster, Older](https://arxiv.org/abs/2409.03137)

    Args:
        learning_rate: A global scaling factor, either fixed or evolving along iterations with a
            scheduler, see :func:`optax.scale_by_learning_rate`.
        b1: Exponential decay rate to track the fast EMA.
        b2: Exponential decay rate to track the second moment of past gradients.
        b3: Exponential decay rate to track the slow EMA.
        alpha: Mixing coefficient in the linear combination fo the fast and slow EMAs.
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
    """

    learning_rate: ScalarOrSchedule
    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.999
    b3: ScalarOrSchedule = 0.9999
    alpha: ScalarOrSchedule = 5.0
    eps: RealNumeric = 1e-08
    eps_root: RealNumeric = 0.0
    mu_dtype: Any = field(default=None, static=True)
    weight_decay: RealNumeric = 0.0
    mask: bool | Weights | Callable[[Weights], Any] | None = field(default=None, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.contrib.ademamix(**as_shallow_dict(self))


@dataclass
class ScaleByAdEMAMix[Weights: PyTree](OptaxWrapper[Weights]):
    """Scale updates according to the Ademamix algorithm.

    References:
        Pagliardini et al., 2024,
            [The AdEMAMix Optimizer: Better, Faster, Older](https://arxiv.org/abs/2409.03137)

    Args:
        b1: Exponential decay rate to track the fast EMA.
        b2: Exponential decay rate to track the second moment of past gradients.
        b3: Exponential decay rate to track the slow EMA.
        alpha: Mixing coefficient in the linear combination for the fast and slow EMAs.
        eps: A small constant applied to denominator outside of the square root (as in the Adam
            paper) to avoid dividing by zero when rescaling.
        eps_root: A small constant applied to denominator inside the square root (as in RMSProp), to
            avoid dividing by zero when rescaling. This is needed for instance when computing
            (meta-)gradients through Adam.
        mu_dtype: Optional `dtype` to be used for the first order accumulator; if `None` then the
            `dtype` is inferred from `params` and `updates`.
    """

    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.999
    b3: ScalarOrSchedule = 0.9999
    alpha: ScalarOrSchedule = 6.0
    eps: RealNumeric = 1e-08
    eps_root: RealNumeric = 0.0
    mu_dtype: Any = field(default=None, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.contrib.scale_by_ademamix(**as_shallow_dict(self))


@dataclass
class SimplifiedAdEMAMix[Weights: PyTree](OptaxWrapper[Weights]):
    """Simplified AdEMAMix.

    References:
        Morwani et al., 2025,
            [Connections between Schedule-Free Optimizers, AdEMAMix, and Accelerated SGD
            Variants](https://arxiv.org/abs/2502.02431)

    Args:
        learning_rate: A global scaling factor, either fixed or evolving along iterations with a
            scheduler, see :func:`optax.scale_by_learning_rate`.
        b1: Exponential decay rate to track the EMA.
        b2: Exponential decay rate to track the second moment of past gradients.
        alpha: Mixing coefficient for the current gradient and EMA.
        eps: A small constant applied to denominator outside of the square root (as in the Adam
            paper) to avoid dividing by zero when rescaling.
        eps_root: A small constant applied to denominator inside the square root (as in RMSProp), to
            avoid dividing by zero when rescaling. This is needed for instance when computing
            (meta-)gradients through Adam.
        weight_decay: Strength of the weight decay regularization. Note that this weight decay is
            multiplied with the learning rate. This is consistent with other frameworks such as
            PyTorch, but different from (Loshchilov et al, 2019) where the weight decay is only
            multiplied with the "schedule multiplier", but not the base learning rate.
        mask: A tree with same structure as (or a prefix of) the params PyTree, or a Callable that
            returns such a pytree given the params/updates. The leaves should be booleans, `True`
            for leaves/subtrees you want to apply the weight decay to, and `False` for those you
            want to skip. Note that the Adam gradient transformations are applied to all parameters.
    """

    learning_rate: ScalarOrSchedule
    b1: RealNumeric = 0.99
    b2: RealNumeric = 0.95
    alpha: ScalarOrSchedule = 0.0
    eps: RealNumeric = 1e-08
    eps_root: RealNumeric = 0.0
    weight_decay: RealNumeric = 0.0
    mask: bool | Weights | Callable[[Weights], Any] | None = field(default=None, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.contrib.simplified_ademamix(**as_shallow_dict(self))


@dataclass
class ScaleBySimplifiedAdEMAMix[Weights: PyTree](OptaxWrapper[Weights]):
    """Scale updates according to the Simplified AdEMAMix optimizer.

    References:
        Morwani et al., 2025,
            [Connections between Schedule-Free Optimizers, AdEMAMix, and Accelerated SGD
            Variants](https://arxiv.org/abs/2502.02431)

    Args:
        b1: Exponential decay rate to track the EMA.
        b2: Exponential decay rate to track the second moment of past gradients.
        alpha: Mixing coefficient for the current gradient and EMA.
        eps: A small constant applied to denominator outside of the square root (as in the Adam
            paper) to avoid dividing by zero when rescaling.
        eps_root: A small constant applied to denominator inside the square root (as in RMSProp), to
            avoid dividing by zero when rescaling. This is needed for instance when computing
            (meta-)gradients through Adam.
    """

    b1: RealNumeric = 0.99
    b2: RealNumeric = 0.95
    alpha: ScalarOrSchedule = 0.0
    eps: RealNumeric = 1e-08
    eps_root: RealNumeric = 0.0

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.contrib.scale_by_simplified_ademamix(**as_shallow_dict(self))


@dataclass
class ADOPT[Weights: PyTree](OptaxWrapper[Weights]):
    """ADOPT (Adaptive Optimization with Provable Theoretical guarantees).

    References:
        Taniguchi et al., 2024,
            [ADOPT: Modified Adam Can Converge with Any β₂ with the Optimal
            Rate](https://arxiv.org/abs/2411.02853)

    Args:
        learning_rate: A global scaling factor, either fixed or evolving along iterations with a
            scheduler.
        b1: Exponential decay rate to track the first moment of past gradients.
        b2: Exponential decay rate to track the second moment of past gradients.
        eps: A small constant applied to denominator outside of the square root to avoid dividing by
            zero when rescaling.
        mu_dtype: Optional `dtype` to be used for the first order accumulator; if `None` then the
            `dtype` is inferred from `params` and `updates`.
        nesterov: Whether to use Nesterov momentum.
        use_clipping: Whether to apply clipping to improve stability. Recommended to keep as True,
            especially for training from scratch.
        clip_value_fn: A function that takes a step index and returns a clipping value. Default is
            :math:`x^{0.25}`.
    """

    learning_rate: ScalarOrSchedule
    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.9999
    eps: RealNumeric = 1e-06
    mu_dtype: Any = field(default=None, static=True)
    _: KW_ONLY
    nesterov: bool = field(default=False, static=True)
    use_clipping: bool = field(default=True, static=True)
    clip_value_fn: Callable[[JaxArray], JaxArray] = field(default=_adopt_clip_value, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.contrib.adopt(**as_shallow_dict(self))


@dataclass
class ScaleByADOPT[Weights: PyTree](OptaxWrapper[Weights]):
    """Rescale updates according to the ADOPT algorithm.

    Args:
        b1: Decay rate for the exponentially weighted average of grads.
        b2: Decay rate for the exponentially weighted average of squared grads.
        eps: Term added to the denominator to improve numerical stability.
        mu_dtype: Optional `dtype` to be used for the first order accumulator; if `None` then the
            `dtype` is inferred from `params` and `updates`.
        nesterov: Whether to use Nesterov momentum.
        use_clipping: Whether to use gradient clipping to improve stability. When enabled, the
            clipping value is determined by the clip_value_fn.
        clip_value_fn: A function that takes a step index and returns a clipping value. Default is
            :math:`x^{0.25}`.
    """

    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.9999
    eps: RealNumeric = 1e-06
    mu_dtype: Any = field(default=None, static=True)
    _: KW_ONLY
    nesterov: bool = field(default=False, static=True)
    use_clipping: bool = field(default=True, static=True)
    clip_value_fn: Callable[[JaxArray], JaxArray] = field(default=_adopt_clip_value, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.contrib.scale_by_adopt(**as_shallow_dict(self))


@dataclass
class COCOB[Weights: PyTree](OptaxWrapper[Weights]):
    """Rescale updates according to the COntinuous COin Betting algorithm.

    References:
        Orabona and Tommasi, 2017,
            [Training Deep Networks without Learning Rates Through Coin
            Betting](https://arxiv.org/abs/1705.07795)

    Args:
        learning_rate: Optional learning rate to e.g. inject some scheduler
        alpha: Fraction to bet parameter of the COCOB optimizer
        eps: Jitter term to avoid dividing by 0
        weight_decay: L2 penalty
        mask: Mask for weight decay
    """

    learning_rate: ScalarOrSchedule = 1.0
    alpha: RealNumeric = 100.0
    eps: RealNumeric = 1e-08
    weight_decay: RealNumeric = 0.0
    mask: bool | Weights | Callable[[Weights], Any] | None = field(default=None, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.contrib.cocob(**as_shallow_dict(self))


@dataclass
class ScaleByCOCOB[Weights: PyTree](OptaxWrapper[Weights]):
    """Rescale updates according to the COntinuous COin Betting algorithm.

    Args:
        alpha: Fraction to bet parameter of the COCOB optimizer
        eps: Jitter term to avoid dividing by 0
    """

    alpha: RealNumeric = 100.0
    eps: RealNumeric = 1e-08

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.contrib.scale_by_cocob(**as_shallow_dict(self))


@dataclass
class DAdaptAdamW[Weights: PyTree](OptaxWrapper[Weights]):
    """Learning rate free AdamW by D-Adaptation.

    References:
        Defazio and Mishchenko, 2023,
            [Learning-Rate-Free Learning by D-Adaptation](https://arxiv.org/abs/2301.07733)

    Args:
        learning_rate: Learning rate scheduling parameter. The recommended schedule is a
            linear_schedule with init_value=1.0 and end_value=0, combined with a 0-20% learning rate
            warmup.
        betas: Betas for the underlying AdamW Optimizer.
        eps: Eps for the underlying AdamW Optimizer.
        estim_lr0: Initial (under-)estimate of the learning rate.
        weight_decay: AdamW style weight-decay. To use Regular Adam decay, chain with
            add_decayed_weights.
    """

    learning_rate: ScalarOrSchedule = 1.0
    betas: tuple[RealNumeric, RealNumeric] = (0.9, 0.999)
    eps: RealNumeric = 1e-08
    estim_lr0: RealNumeric = 1e-06
    weight_decay: RealNumeric = 0.0

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.contrib.dadapt_adamw(**as_shallow_dict(self))


@dataclass
class DifferentiallyPrivateAggregate[Weights: PyTree](OptaxWrapper[Weights]):
    """Aggregates gradients based on the DPSGD algorithm.

    References:
        Abadi et al., 2016,
            [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133)

    Args:
        l2_norm_clip: Maximum L2 norm of the per-example gradients.
        noise_multiplier: Ratio of standard deviation to the clipping norm.
        key: Random generator key for noise generation.
    """

    l2_norm_clip: RealNumeric
    noise_multiplier: RealNumeric
    key: KeyArray | int | None = None

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.contrib.differentially_private_aggregate(**as_shallow_dict(self))


@dataclass
class DoG[Weights: PyTree](OptaxWrapper[Weights]):
    r"""Distance over Gradients (DoG) optimizer.

    References:
        Ivgi et al., 2023,
            [DoG is SGD's Best Friend: A Parameter-Free Dynamic Step Size
            Schedule](https://arxiv.org/abs/2302.12022)

    Args:
        learning_rate: Optional learning rate (potentially varying according to some predetermined
            scheduler).
        init_step: Initial step specification. Consists of a pair ``(tag, value)``, where ``value``
            is a float and ``tag`` is a string, which must be one of ``distance``,
            ``learning_rate``, or ``heuristic``. ``distance`` sets the initial distance :math:`r_0`
            (:math:`r_\epsilon` in the paper) to the given value. ``learning_rate`` sets the initial
            learning rate :math:`\eta_0` to the given value. ``heuristic`` sets :math:`r_0 = \alpha
            (1 + \|x_0\|)`, where :math:`\alpha` is the given value. The suggested value of
            :math:`\alpha` is 1e-6, unless the model uses batch normalization, in which case the
            suggested value is 1e-4. As discussed in the paper, the value should be small enough to
            ensure that the initial update step will be small enough to not cause the model to
            diverge.
        eps: Epsilon used for numerical stability - added to the sum of squared norm of gradients.
        weight_decay: Strength of the weight decay regularization.
        mask: A tree with same structure as (or a prefix of) the params PyTree, or a Callable that
            returns such a pytree given the params/updates. The leaves should be booleans, `True`
            for leaves/subtrees you want to apply the weight decay to, and `False` for those you
            want to skip. Note that the gradient transformations is applied to all parameters.
    """

    learning_rate: ScalarOrSchedule = 1.0
    init_step: tuple[Literal["distance", "learning_rate", "heuristic"], RealNumeric] = field(
        default=("heuristic", 1e-06), static=True
    )
    eps: RealNumeric = 1e-08
    weight_decay: RealNumeric | None = None
    mask: bool | Weights | Callable[[Weights], Any] | None = field(default=None, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.contrib.dog(**as_shallow_dict(self))


@dataclass
class DoWG[Weights: PyTree](OptaxWrapper[Weights]):
    """Distance over weighted Gradients optimizer.

    References:
        Khaled et al., 2023,
            [DoWG Unleashed: An Efficient Universal Parameter-Free Gradient Descent
            Method](https://arxiv.org/abs/2305.16284)

    Args:
        learning_rate: Optional learning rate (potentially varying according to some predetermined
            scheduler).
        init_estim_sq_dist: Initial guess of the squared distance to solution.
        eps: Small value to prevent division by zero in the denominator defining, the learning rate,
            also used as initial guess for the distance to solution if ``init_estim_sq_dist`` is
            None.
        weight_decay: Strength of the weight decay regularization.
        mask: A tree with same structure as (or a prefix of) the params PyTree, or a Callable that
            returns such a pytree given the params/updates. The leaves should be booleans, `True`
            for leaves/subtrees you want to apply the weight decay to, and `False` for those you
            want to skip. Note that the gradient transformations is applied to all parameters.
    """

    learning_rate: ScalarOrSchedule = 1.0
    init_estim_sq_dist: RealNumeric | None = None
    eps: RealNumeric = 0.0001
    weight_decay: RealNumeric | None = None
    mask: bool | Weights | Callable[[Weights], Any] | None = field(default=None, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.contrib.dowg(**as_shallow_dict(self))


@dataclass
class DPSGD[Weights: PyTree](OptaxWrapper[Weights]):
    """Pytree-compatible wrapper for the optax DP-SGD optimiser.

    Differentially Private SGD clips per-example gradients and adds Gaussian
    noise to provide (ε, δ)-differential privacy guarantees.

    References:
        Abadi et al., 2016,
            [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133)

    Args:
        learning_rate: Global step-size schedule or scalar.
        l2_norm_clip: Per-example gradient clipping threshold.
        noise_multiplier: Ratio of noise standard deviation to ``l2_norm_clip``.
        seed: Seed for the noise RNG.
        momentum: Optional momentum decay rate.
        nesterov: Whether to use Nesterov momentum.
    """

    learning_rate: ScalarOrSchedule
    l2_norm_clip: RealNumeric
    noise_multiplier: RealNumeric
    seed: IntegralNumeric
    momentum: float | None = None
    nesterov: bool = field(default=False, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.contrib.dpsgd(**as_shallow_dict(self))


@dataclass
class GaLore[Weights: PyTree](OptaxWrapper[Weights]):
    """GaLore: Memory-efficient training via gradient lowrank projection.

    References:
        Zhao et al., 2024,
            [GaLore: Memory-Efficient LLM Training by Gradient Low-Rank
            Projection](https://arxiv.org/abs/2403.03507)

    Args:
        learning_rate: A global scaling factor, either fixed or evolving along iterations with a
            scheduler.
        rank: Target rank for the low-rank projection. Lower values save more memory but may slow
            convergence. Default 128 is a good starting point.
        update_proj_gap: Number of steps between projection matrix updates. The projectors are
            recomputed from the gradient SVD every this many steps to adapt to the changing gradient
            landscape.
        scale: Additional scaling factor for updates.
        base_optimizer: The base gradient transformation to apply in the low-rank subspace. Must be
            a gradient-only transformation like ``scale_by_adam``, NOT an optimizer requiring params
            like ``adamw``. If None, defaults to ``optax.scale_by_adam()``. If the base optimizer
            includes a learning rate, set ``learning_rate=1.0`` here to avoid double-scaling.
        weight_decay: Strength of decoupled weight decay regularization (as in AdamW). This is
            applied correctly in full parameter space, unlike weight decay in the base optimizer
            which would fail.
        mask: A tree with same structure as params PyTree, or a Callable that returns such a pytree.
            Leaves should be booleans indicating whether to apply weight decay to each parameter.
        weight_dimension_numbers: Specifies how to treat non-2D tensors as 2D matrices for
            projection. See ``scale_by_galore`` for details.
    """

    learning_rate: ScalarOrSchedule
    rank: int = field(default=128, static=True)
    update_proj_gap: int = field(default=200, static=True)
    scale: RealNumeric = 1.0
    base_optimizer: GradientTransformation[Any, Weights] | None = None
    weight_decay: RealNumeric = 0.0
    mask: bool | Weights | Callable[[Weights], Any] | None = field(default=None, static=True)
    weight_dimension_numbers: PyTree | Callable[[Weights], PyTree] = field(
        default=None, static=True
    )

    @override
    def _optax(self) -> OptaxGradientTransformation:
        kwargs = as_shallow_dict(self)
        kwargs["base_optimizer"] = to_optax_or_none(self.base_optimizer)
        return optax.contrib.galore(**kwargs)


@dataclass
class ScaleByGaLore[Weights: PyTree](OptaxWrapper[Weights]):
    """Scale updates using GaLore (Gradient Low-Rank Projection).

    References:
        Zhao et al., 2024,
            [GaLore: Memory-Efficient LLM Training by Gradient Low-Rank
            Projection](https://arxiv.org/abs/2403.03507)

    Args:
        rank: Target rank for the low-rank projection. Lower rank = less memory but potentially
            slower convergence.
        update_proj_gap: Number of steps between projection matrix updates. The projection matrices
            are recomputed from gradient SVD every this many steps.
        scale: Scaling factor applied to the final updates.
        base_optimizer: The base gradient transformation to apply in the low-rank subspace for 2D
            params and full space for non-2D params. Must be a gradient-only transformation (see
            warning above). If None, defaults to ``transform.scale_by_adam()``.
        weight_dimension_numbers: Specifies how to treat non-2D tensors as 2D matrices. Can be: -
            None: Only project naturally 2D parameters (default behavior) - A single
            ``GaLoreDimensionNumbers``: Apply to all parameters - A pytree matching params structure
            with ``GaLoreDimensionNumbers`` at leaves (use None for params to skip) - A callable
            taking params and returning such a pytree
    """

    rank: int = field(default=128, static=True)
    update_proj_gap: int = field(default=200, static=True)
    scale: RealNumeric = 1.0
    base_optimizer: GradientTransformation[Any, Weights] | None = None
    weight_dimension_numbers: PyTree | Callable[[Weights], PyTree] = field(
        default=None, static=True
    )

    @override
    def _optax(self) -> OptaxGradientTransformation:
        kwargs = as_shallow_dict(self)
        kwargs["base_optimizer"] = to_optax_or_none(self.base_optimizer)
        return optax.contrib.scale_by_galore(**kwargs)


@dataclass
class HutchinsonEstimatorDiagHessian[Weights: PyTree](OptaxWrapper[Weights]):
    """Returns a GradientTransformationExtraArgs computing the Hessian diagonal.

    Args:
        random_seed: Key used to generate random vectors.
    """

    random_seed: KeyArray | None = None

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.contrib.hutchinson_estimator_diag_hessian(**as_shallow_dict(self))


@dataclass
class MADGRAD[Weights: PyTree](OptaxWrapper[Weights]):
    """The MADGRAD optimizer.

    References:
        Defazio and Jelassi, 2021,
            [Adaptivity without Compromise: A Momentumized, Adaptive, Dual Averaged Gradient Method
            for Stochastic Optimization](https://arxiv.org/abs/2101.11075)

    Args:
        learning_rate: A global scaling factor, either fixed or evolving along iterations with a
            scheduler.
        momentum: Momentum parameter (default: 0.9).
        weight_decay: Strength of the weight decay regularization (L2).
        eps: Term added to the denominator to improve numerical stability.
    """

    learning_rate: ScalarOrSchedule
    momentum: RealNumeric = 0.9
    weight_decay: RealNumeric = 0.0
    eps: RealNumeric = 1e-06

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.contrib.madgrad(**as_shallow_dict(self))


@dataclass
class ScaleByMADGRAD[Weights: PyTree](OptaxWrapper[Weights]):
    """Rescale updates according to the MADGRAD algorithm.

    References:
        Defazio and Jelassi, 2021,
            [Adaptivity without Compromise: A Momentumized, Adaptive, Dual Averaged Gradient Method
            for Stochastic Optimization](https://arxiv.org/abs/2101.11075)

    Args:
        learning_rate: A global scaling factor, either fixed or evolving along iterations with a
            scheduler.
        momentum: Momentum parameter (default: 0.9).
        eps: Term added to the denominator to improve numerical stability.
    """

    learning_rate: ScalarOrSchedule
    momentum: RealNumeric = 0.9
    eps: RealNumeric = 1e-06

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.contrib.scale_by_madgrad(**as_shallow_dict(self))


@dataclass
class Mechanize[Weights: PyTree](OptaxWrapper[Weights]):
    """Mechanic - a black box learning rate tuner/optimizer.

    References:
        Cutkosky et al., 2023, [Mechanic: A Learning Rate Tuner](https://arxiv.org/abs/2306.00144)

    Args:
        base_optimizer: Base optimizer to compute updates from.
        weight_decay: A scalar weight decay rate. Note that this weight decay is not the same as the
            weight decay one would use for the base_optimizer. In addition to sometimes helping
            converge faster, this helps Mechanic reduce the variance between training runs using
            different seeds. You likely would not need to tune this, the default should work in most
            cases.
        eps: Epsilon for mechanic.
        s_init: Initial scale factor. Default should work almost all the time.
        num_betas: Unlike traditional exp accumulators (like 1st or 2nd moment of adam), where one
            has to choose an explicit beta, mechanic has a clever way to automatically learn the
            right beta for all accumulators. We only provide the range of possible betas, and not
            the tuned value. For instance, if you set num_betas to 3, it will use betas = [0.9,
            0.99, 0.999].
    """

    base_optimizer: GradientTransformation[Any, Weights]
    weight_decay: RealNumeric = 0.01
    eps: RealNumeric = 1e-08
    s_init: RealNumeric = 1e-06
    num_betas: int = field(default=6, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        kwargs = as_shallow_dict(self)
        kwargs["base_optimizer"] = to_optax_or_none(self.base_optimizer)
        return optax.contrib.mechanize(**kwargs)


@dataclass
class Momo[Weights: PyTree](OptaxWrapper[Weights]):
    """Adaptive Learning Rates for SGD with momentum.

    References:
        Schaipp et al., 2023,
            [MoMo: Momentum Models for Adaptive Learning Rates](https://arxiv.org/abs/2305.07583)

    Args:
        learning_rate: User-specified learning rate. Recommended to be chosen rather large, by
            default 1.0.
        beta: Momentum coefficient (for EMA).
        lower_bound: Lower bound of the loss. Zero should be a good choice for many tasks.
        weight_decay: Weight-decay parameter.
        adapt_lower_bound: If no good guess for the lower bound is available, set this to true, in
            order to estimate the lower bound on the fly (see the paper for details).
    """

    learning_rate: ScalarOrSchedule = 1.0
    beta: RealNumeric = 0.9
    lower_bound: RealNumeric = 0.0
    weight_decay: RealNumeric = 0.0
    adapt_lower_bound: bool = field(default=False, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.contrib.momo(**as_shallow_dict(self))


@dataclass
class MomoAdam[Weights: PyTree](OptaxWrapper[Weights]):
    """Adaptive Learning Rates for Adam(W).

    References:
        Schaipp et al., 2023,
            [MoMo: Momentum Models for Adaptive Learning Rates](https://arxiv.org/abs/2305.07583)

    Args:
        learning_rate: User-specified learning rate. Recommended to be chosen rather large, by
            default 1.0.
        b1: Exponential decay rate to track the first moment of past gradients.
        b2: Exponential decay rate to track the second moment of past gradients.
        eps: Eps for the underlying Adam Optimizer.
        lower_bound: Lower bound of the loss. Zero should be a good choice for many tasks.
        weight_decay: Weight-decay parameter. Momo-Adam performs weight decay in similar fashion to
            AdamW.
        adapt_lower_bound: If no good guess for the lower bound is available, set this to true, in
            order to estimate the lower bound on the fly (see the paper for details).
    """

    learning_rate: ScalarOrSchedule = 0.01
    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.999
    eps: RealNumeric = 1e-08
    lower_bound: RealNumeric = 0.0
    weight_decay: RealNumeric = 0.0
    adapt_lower_bound: bool = field(default=False, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.contrib.momo_adam(**as_shallow_dict(self))


@dataclass
class Muon[Weights: PyTree](OptaxWrapper[Weights]):
    """Muon: Momentum Orthogonalized by Newton-schulz.

    References:
        Jordan, 2024,
            [modded-nanogpt: Speedrunning the NanoGPT
            baseline](https://github.com/KellerJordan/modded-nanogpt)
        Bernstein and Newhouse, 2024,
            [Old Optimizer, New Norm: An Anthology](https://arxiv.org/abs/2409.20325)
        Liu et al., 2025, [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982)
        Boissin et al., 2025,
            [Turbo-Muon: Almost-Orthogonal Pre-Conditioning for Fast Muon
            Updates](https://arxiv.org/abs/2512.04632)
        Ahn et al., 2025,
            [Dion: Distributed Orthonormalized Updates](https://arxiv.org/abs/2504.05295)
        Grishina et al., 2025,
            [Accelerating Newton-Schulz Iteration for Orthogonalization via Chebyshev-type
            Polynomials](https://arxiv.org/abs/2506.10935)
        Amsel et al., 2025,
            [The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon
            Algorithm](https://arxiv.org/abs/2505.16932)

    Args:
        learning_rate: A global scaling factor, either fixed or evolving along iterations with a
            scheduler, see :func:`optax.scale_by_learning_rate`.
        ns_coeffs: Coefficients for the Newton-schulz method (can be a string indicator for a
            preset). Existing presets: `muon`, `dion`.
        ns_steps: Number of Newton-schulz iterations. Ignored if `ns_coeffs` is a tuple of tuples.
        beta: Decay rate for the exponentially weighted average of grads.
        eps: Term added to the denominator to improve numerical stability.
        weight_decay: Strength of the weight decay regularization. Note that this weight decay is
            multiplied with the learning rate. This is consistent with other frameworks such as
            PyTorch, but different from (Loshchilov et al, 2019) where the weight decay is only
            multiplied with the "schedule multiplier", but not the base learning rate.
        weight_decay_mask: A tree with same structure as (or a prefix of) the params PyTree, or a
            Callable that returns such a pytree given the params/updates. The leaves should be
            booleans, `True` for leaves/subtrees you want to apply the weight decay to, and `False`
            for those you want to skip.
        mu_dtype: Data type of the momentum accumulator.
        nesterov: Whether to use Nesterov momentum.
        adaptive: Whether to scale the updates by the dual norm of the original updates
            ([Bernstein and Newhouse, 2024](https://arxiv.org/abs/2409.20325)).
        preconditioning: The preconditioning to apply before the Newton-Schulz iterations:

            - `"frobenius"` (default): Frobenius rescaling.  It is safe and standard, but degrades
              orthogonalization when using fewer than 5 Newton-Schulz steps.
            - `"spectral"`: Spectral-norm rescaling.  It costs much more computation but
              orthogonalizes better.
            - `"aol"`: AOL rescaling, which improves orthogonality with little overhead and usually
              saves one Newton-Schulz step ([Boissin et al., 2025](https://arxiv.org/abs/2512.04632)).
            - `"schatten"`: Schatten-4-norm rescaling, which improves performance at little extra
              cost ([Grishina et al., 2025](https://arxiv.org/abs/2506.10935)).
        adam_b1: Exponential decay rate for Adam's first moment estimates.
        adam_b2: Exponential decay rate for Adam's second moment estimates.
        adam_eps_root: Epsilon to stabilize division in Adam, square root version.
        adam_weight_decay: Weight decay factor for Adam.
        adam_learning_rate: Auxiliary learning rate for the Adam optimizer. If `None`, the learning
            rate for Adam defaults to the same as Muon.
        muon_weight_dimension_numbers: An optional tree of `MuonDimensionNumbers`s, specifying how
            to reshape the parameters for orthogonalization otherwise muon parameters are assumed to
            be 2D matrices. A `None` value indicates that the parameter is not a muon parameter and
            will be optimized with Adam. A callable takes as input the params and returns a possibly
            masked pytree of specs, similar to `weight_decay_mask`. If not provided, muon is applied
            to all 2D parameters.
        consistent_rms: An optional float to activate consistent RMS scaling. Scales updates by
            `sqrt(max(fan_in, fan_out)) * consistent_rms` to make root mean square (RMS)
            shape-independent, like AdamW ([Liu et al., 2025](https://arxiv.org/abs/2502.16982)).
            `0.2` is recommended to match AdamW's empirical RMS. If `None`, uses width scaling
            `sqrt(max(1, fan_out / fan_in))`.
    """

    learning_rate: ScalarOrSchedule
    ns_coeffs: tuple[float, ...] | tuple[tuple[float, ...], ...] | str = field(
        default=(3.4445, -4.775, 2.0315), static=True
    )
    ns_steps: int = field(default=5, static=True)
    beta: RealNumeric = 0.95
    eps: RealNumeric = 1e-08
    weight_decay: RealNumeric = 0.0
    weight_decay_mask: bool | Weights | Callable[[Weights], Any] | None = field(
        default=None, static=True
    )
    mu_dtype: Any = field(default=None, static=True)
    _: KW_ONLY
    nesterov: bool = field(default=True, static=True)
    adaptive: bool = field(default=False, static=True)
    preconditioning: Literal["frobenius", "spectral", "aol", "schatten"] = field(
        default="frobenius", static=True
    )
    adam_b1: RealNumeric = 0.9
    adam_b2: RealNumeric = 0.999
    adam_eps_root: RealNumeric = 0.0
    adam_weight_decay: RealNumeric = 0.0
    adam_learning_rate: ScalarOrSchedule | None = None
    muon_weight_dimension_numbers: PyTree | Callable[[Weights], PyTree] = field(
        default=None, static=True
    )
    consistent_rms: RealNumeric | None = None

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.contrib.muon(**as_shallow_dict(self))


@dataclass
class ScaleByMuon[Weights: PyTree](OptaxWrapper[Weights]):
    """Rescale updates according to the Muon algorithm.

    References:
        Jordan, 2024,
            [modded-nanogpt: Speedrunning the NanoGPT
            baseline](https://github.com/KellerJordan/modded-nanogpt)
        Bernstein and Newhouse, 2024,
            [Old Optimizer, New Norm: An Anthology](https://arxiv.org/abs/2409.20325)
        Liu et al., 2025, [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982)
        Boissin et al., 2025,
            [Turbo-Muon: Almost-Orthogonal Pre-Conditioning for Fast Muon
            Updates](https://arxiv.org/abs/2512.04632)
        Ahn et al., 2025,
            [Dion: Distributed Orthonormalized Updates](https://arxiv.org/abs/2504.05295)
        Grishina et al., 2025,
            [Accelerating Newton-Schulz Iteration for Orthogonalization via Chebyshev-type
            Polynomials](https://arxiv.org/abs/2506.10935)
        Amsel et al., 2025,
            [The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon
            Algorithm](https://arxiv.org/abs/2505.16932)

    Args:
        ns_coeffs: Coefficients for the Newton-schulz method.
        ns_steps: Number of Newton-schulz iterations. Ignored if `ns_coeffs` is a tuple of tuples.
        beta: Decay rate for the exponentially weighted average of grads.
        eps: Term added to denominators to improve numerical stability.
        mu_dtype: Data type of the momentum accumulator.
        nesterov: Whether to use Nesterov momentum.
        adaptive: Whether to scale the updates by the dual norm of the original updates
            ([Bernstein and Newhouse, 2024](https://arxiv.org/abs/2409.20325)).
        preconditioning: The preconditioning to apply before the Newton-Schulz iterations:

            - `"frobenius"` (default): Frobenius rescaling.
            - `"spectral"`: Spectral-norm rescaling.
            - `"aol"`: AOL rescaling, which improves orthogonality.
            - `"schatten"`: Schatten-4-norm rescaling.
        weight_dimension_numbers: An optional tree with the same structure as the params of
            `MuonDimensionNumbers`s, specifying how to reshape the parameters before and after the
            orthogonalization OR a callable returning such a tree. None implies that all parameters
            are 2D matrices.
    """

    ns_coeffs: tuple[float, ...] | tuple[tuple[float, ...], ...] = field(
        default=(3.4445, -4.775, 2.0315), static=True
    )
    ns_steps: int = field(default=5, static=True)
    beta: RealNumeric = 0.95
    eps: RealNumeric = 1e-08
    mu_dtype: Any = field(default=None, static=True)
    _: KW_ONLY
    nesterov: bool = field(default=True, static=True)
    adaptive: bool = field(default=False, static=True)
    preconditioning: Literal["frobenius", "spectral", "aol", "schatten"] = field(
        default="frobenius", static=True
    )
    weight_dimension_numbers: PyTree | Callable[[Weights], PyTree] = field(
        default=None, static=True
    )

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.contrib.scale_by_muon(**as_shallow_dict(self))


@dataclass
class Normalize[Weights: PyTree](OptaxWrapper[Weights]):
    """Normalizes the gradient."""

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.contrib.normalize(**as_shallow_dict(self))


@dataclass
class Prodigy[Weights: PyTree](OptaxWrapper[Weights]):
    """Learning rate free AdamW with Prodigy.

    References:
        Mishchenko and Defazio, 2023,
            [Prodigy: An Expeditiously Adaptive Parameter-Free
            Learner](https://arxiv.org/abs/2306.06101)

    Args:
        learning_rate: Learning rate scheduling parameter. The recommended schedule is a
            linear_schedule with init_value=1.0 and end_value=0, combined with a 0-20% learning rate
            warmup.
        betas: Betas for the underlying AdamW Optimizer.
        beta3: Optional momentum parameter for estimation of D.
        eps: Eps for the underlying AdamW Optimizer.
        estim_lr0: Initial (under-)estimate of the learning rate.
        estim_lr_coef: LR estimates are multiplied by this parameter.
        weight_decay: AdamW style weight-decay. To use Regular Adam decay, chain with
            add_decayed_weights.
        safeguard_warmup: Remove lr from the denominator of D estimate to avoid issues during
            warm-up stage. Off by default.
    """

    learning_rate: ScalarOrSchedule = 1.0
    betas: tuple[RealNumeric, RealNumeric] = (0.9, 0.999)
    beta3: RealNumeric | None = None
    eps: RealNumeric = 1e-08
    estim_lr0: RealNumeric = 1e-06
    estim_lr_coef: RealNumeric = 1.0
    weight_decay: RealNumeric = 0.0
    safeguard_warmup: bool = field(default=False, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.contrib.prodigy(**as_shallow_dict(self))


@dataclass
class ReduceOnPlateau[Weights: PyTree](OptaxWrapper[Weights]):
    """Reduce learning rate when a metric has stopped improving.

    Args:
        factor: Factor by which to reduce the learning rate. new_scale = scale * factor.
        patience: Number of iterations with no improvement after which learning rate will be
            reduced.
        rtol: Relative tolerance for measuring new optimum.
        atol: Absolute tolerance for measuring new optimum.
        cooldown: Number of iterations to wait before resuming normal operation after scale has been
            reduced.
        accumulation_size: Number of values to aggregate before applying the logic of reduce on
            plateau. If the value fed to the optimizer is a test value, simply take 1 (default). If
            the value fed to the optimizer is the loss on a the current minibatch, consider using a
            larger accumulation size.
        min_scale: Scale at which the learning rate decay stops.
    """

    factor: float = field(default=0.1, static=True)
    patience: IntegralNumeric = 10
    rtol: float = field(default=0.0001, static=True)
    atol: float = field(default=0.0, static=True)
    cooldown: IntegralNumeric = 0
    accumulation_size: IntegralNumeric = 1
    min_scale: RealNumeric = 0.0

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.contrib.reduce_on_plateau(**as_shallow_dict(self))


@dataclass
class SAM[Weights: PyTree](OptaxWrapper[Weights]):
    """Implementation of SAM (Sharpness Aware Minimization).

    References:
        Foret et al., 2020,
            [Sharpness-Aware Minimization for Efficiently Improving
            Generalization](https://arxiv.org/abs/2010.01412)

    Args:
        optimizer: The outer optimizer.
        adv_optimizer: The inner adversarial optimizer.
        sync_period: How often to run the outer optimizer, defaults to 2, or every other step.
        reset_state: Whether to reset the state of the inner optimizer after every sync period,
            defaults to ``True``.
        opaque_mode: Bool. If ``True``, the outer optimizer and the adversarial optimizer are run in
            an internal loop at each call to ``update``, so that adversarial updates are opaque to
            the rest of the system. If ``False``, one optimizer is (effectively) evaluated per call
            to ``update``, meaning that adversarial updates are visible to the rest of the system.
            Setting ``opaque_mode`` to ``True`` is necessary if the training system using SAM has
            side effects from each call to ``update`` besides the changes to the model's parameters.
            The most common example would be if the model uses BatchNorm statistics -- those
            statistics would be updated on both adversarial and non-adversarial update steps,
            causing them to get out of sync with the model's parameters (which are effectively only
            updated on non-adversarial steps). See the NOTE section for more details on
            ``opaque_mode=True``.
        batch_axis_name: Str or None. Only used if ``opaque_mode=True``. When running in a pmapped
            setting, it is necessary to take a ``jax.lax.pmean`` of the adversarial updates
            internally before passing them to the outer optimizer. You only need to specify this if
            you have to use ``jax.lax.pmean`` in your training loop.
    """

    optimizer: GradientTransformation[Any, Weights]
    adv_optimizer: GradientTransformation[Any, Weights]
    sync_period: int = field(default=2, static=True)
    reset_state: bool = field(default=True, static=True)
    opaque_mode: bool = field(default=False, static=True)
    batch_axis_name: str | None = field(default=None, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        kwargs = as_shallow_dict(self)
        kwargs["optimizer"] = to_optax_or_none(self.optimizer)
        kwargs["adv_optimizer"] = to_optax_or_none(self.adv_optimizer)
        return optax.contrib.sam(**kwargs)


@dataclass
class ScheduleFree[Weights: PyTree](OptaxWrapper[Weights]):
    """Turn base_optimizer schedule_free.

    References:
        Defazio et al., 2024, [The Road Less Scheduled](https://arxiv.org/abs/2405.15682)
        Defazio et al., 2024,
            [Schedule-Free Learning: A New Way to
            Train](https://github.com/facebookresearch/schedule_free)

    Args:
        base_optimizer: Base optimizer to compute updates from.
        learning_rate: Learning_rate schedule w/o decay but with warmup.
        b1: Beta_1 parameter in the y update.
        weight_lr_power: We downweight the weight of averaging using this. This is especially
            helpful in early iterations during warmup.
        state_dtype: Dtype for z sequence in the schedule free method.
    """

    base_optimizer: GradientTransformation[Any, Weights]
    learning_rate: ScalarOrSchedule
    b1: RealNumeric = 0.9
    weight_lr_power: RealNumeric = 2.0
    state_dtype: Any = field(default=None, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        kwargs = as_shallow_dict(self)
        kwargs["base_optimizer"] = to_optax_or_none(self.base_optimizer)
        return optax.contrib.schedule_free(**kwargs)


@dataclass
class ScheduleFreeAdamW[Weights: PyTree](OptaxWrapper[Weights]):
    """Schedule-Free wrapper for AdamW.

    Args:
        learning_rate: AdamW learning rate.
        warmup_steps: Positive integer, the length of the linear warmup.
        b1: Beta_1 parameter in the y update.
        b2: Exponential decay rate to track the second moment of past gradients.
        eps: A small constant applied to denominator outside of the square root (as in the Adam
            paper) to avoid dividing by zero when rescaling.
        weight_decay: Strength of the weight decay regularization.
        weight_lr_power: We downweight the weight of averaging using this. This is especially
            helpful in early iterations during warmup.
        state_dtype: Dtype for z sequence in the schedule free method.
    """

    learning_rate: RealNumeric = 0.0025
    warmup_steps: int | None = field(default=None, static=True)
    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.999
    eps: RealNumeric = 1e-08
    weight_decay: RealNumeric = 0.0
    weight_lr_power: RealNumeric = 2.0
    state_dtype: Any = field(default=None, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.contrib.schedule_free_adamw(**as_shallow_dict(self))


@dataclass
class ScheduleFreeSGD[Weights: PyTree](OptaxWrapper[Weights]):
    """Schedule-Free wrapper for SGD.

    Args:
        learning_rate: SGD learning rate.
        warmup_steps: Positive integer, the length of the linear warmup.
        b1: Beta_1 parameter in the y update.
        weight_decay: Strength of the weight decay regularization. Note that this weight decay is
            multiplied with the learning rate. This is consistent with other frameworks such as
            PyTorch, but different from (Loshchilov et al, 2019) where the weight decay is only
            multiplied with the "schedule multiplier", but not the base learning rate.
        weight_lr_power: We downweight the weight of averaging using this. This is especially
            helpful in early iterations during warmup.
        state_dtype: Dtype for z sequence in the schedule free method.
    """

    learning_rate: RealNumeric = 1.0
    warmup_steps: int | None = field(default=None, static=True)
    b1: RealNumeric = 0.9
    weight_decay: RealNumeric | None = None
    weight_lr_power: RealNumeric = 2.0
    state_dtype: Any = field(default=None, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.contrib.schedule_free_sgd(**as_shallow_dict(self))


@dataclass
class Sophia[Weights: PyTree](OptaxWrapper[Weights]):
    """Sophia optimizer.

    References:
        Liu et al., 2023,
            [Sophia: A Scalable Stochastic Second-order Optimizer for Language Model
            Pre-training](https://arxiv.org/abs/2305.14342)
        [Levanter](https://github.com/marin-community/levanter)

    Args:
        learning_rate: A global scaling factor, either fixed or evolving along iterations with a
            scheduler, see :func:`optax.scale_by_learning_rate`.
        b1: Exponential decay rate for the first moment estimates.
        b2: Exponential decay rate for the hessian diagonal estimates. Keep in mind effective `b2`
            is `1 - (1 - b2) / update_interval`, e.g. default `b2` of 0.99 is effectively 0.999
            because default `update_interval` is every 10.
        eps: Small constant to avoid division by zero.
        weight_decay: Rate at which to decay weights.
        weight_decay_mask: A tree with same structure as (or a prefix of) the params PyTree, or a
            Callable that returns such a pytree given the params/updates. The leaves should be
            booleans, `True` for leaves/subtrees you want to apply the transformation to, and
            `False` for those you want to skip.
        gamma: Normalizing constant for the hessian diagonal.
        clip_threshold: Threshold for clipping updates.
        update_interval: Interval for updating the hessian diagonal.
        hessian_diagonal_fn: Transformation that computes the diagonal of the Hessian. Default is
            ``HutchinsonEstimatorDiagHessian`` (sophia-h). If using more than one device, be sure
            this function properly averages the hessian diagonal across devices.
        mu_dtype: Dtype of the first moment estimates.
        verbose: If True, print win rate every n steps.
        print_win_rate_every_n_steps: Print sophia win rate every n steps for diagnostic purposes.
            Authors state this value should stay between 0.1 and 0.5 during training. If win rate is
            too low, try increasing `gamma`. 0 to turn off.
    """

    learning_rate: ScalarOrSchedule
    b1: RealNumeric = 0.965
    b2: RealNumeric = 0.99
    eps: RealNumeric = 1e-08
    weight_decay: RealNumeric = 0.0001
    weight_decay_mask: bool | Weights | Callable[[Weights], Any] | None = field(
        default=None, static=True
    )
    gamma: RealNumeric = 0.01
    clip_threshold: RealNumeric | None = 1.0
    update_interval: RealNumeric = 10
    hessian_diagonal_fn: GradientTransformation[Any, Weights] = field(
        default_factory=HutchinsonEstimatorDiagHessian
    )
    mu_dtype: Any = field(default=None, static=True)
    verbose: bool = field(default=False, static=True)
    print_win_rate_every_n_steps: int = field(default=0, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        kwargs = as_shallow_dict(self)
        kwargs["hessian_diagonal_fn"] = to_optax_or_none(self.hessian_diagonal_fn)
        return optax.contrib.sophia(**kwargs)


@dataclass
class SplitRealAndImaginary[Weights: PyTree](OptaxWrapper[Weights]):
    """Splits the real and imaginary components of complex updates into two.

    Args:
        inner: The inner transformation.
    """

    inner: GradientTransformation[Any, Weights]

    @override
    def _optax(self) -> OptaxGradientTransformation:
        kwargs = as_shallow_dict(self)
        kwargs["inner"] = to_optax_or_none(self.inner)
        return optax.contrib.split_real_and_imaginary(**kwargs)
