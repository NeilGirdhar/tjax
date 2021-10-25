from typing import Any, Callable, Generic, Optional, Tuple, Union

import jax.numpy as jnp
from optax import AddNoiseState
from optax import ApplyEvery as ApplyEveryState
from optax import (EmaState, EmptyState, ScaleByAdamState, ScaleByRmsState, ScaleByRssState,
                   ScaleByRStdDevState, ScaleByScheduleState, ScaleBySM3State,
                   ScaleByTrustRatioState, ScaleState, Schedule, TraceState, add_decayed_weights,
                   add_noise, apply_every, centralize, ema, scale, scale_by_adam, scale_by_belief,
                   scale_by_param_block_norm, scale_by_param_block_rms, scale_by_radam,
                   scale_by_rms, scale_by_rss, scale_by_schedule, scale_by_sm3, scale_by_stddev,
                   scale_by_trust_ratio, scale_by_yogi, trace)
from optax._src.transform import ScaleByBeliefState  # type: ignore

from ..annotations import RealNumeric
from ..dataclasses import dataclass, field
from ..generator import Generator
from .transform import GradientTransformation, Weights

__all__ = ['Trace', 'Ema', 'ScaleByRss', 'ScaleByRms', 'ScaleByStddev', 'ScaleByAdam', 'Scale',
           'ScaleByParamBlockNorm', 'ScaleByParamBlockRMS', 'ScaleByBelief', 'ScaleByYogi',
           'ScaleByRAdam', 'AddDecayedWeights', 'ScaleBySchedule', 'ScaleByTrustRatio', 'AddNoise',
           'ApplyEvery', 'Centralize']


@dataclass
class Trace(GradientTransformation[TraceState, Weights], Generic[Weights]):
    """Compute a trace of past updates.

    Note: `trace` and `ema` have very similar but distinct updates;
    `trace = decay * trace + t`, while `ema = decay * ema + (1-decay) * t`.
    Both are frequently found in the optimisation literature.

    Args:
        decay: the decay rate for the trace of past updates.
        nesterov: whether to use Nesterov momentum.
        accumulator_dtype: optional `dtype` to be used for the accumulator; if
            `None` then the `dtype` is inferred from `params` and `updates`.
    """
    decay: RealNumeric
    nesterov: bool = field(default=False, static=True)
    accumulator_dtype: Any = field(default=None, static=True)

    def init(self, parameters: Weights) -> TraceState:
        return trace(self.decay, self.nesterov, self.accumulator_dtype).init(parameters)

    def update(self,
               gradient: Weights,
               state: TraceState,
               parameters: Optional[Weights]) -> Tuple[Weights, TraceState]:
        return trace(self.decay, self.nesterov, self.accumulator_dtype).update(gradient, state,
                                                                               parameters)


@dataclass
class Ema(GradientTransformation[EmaState, Weights], Generic[Weights]):
    """Compute an exponential moving average of past updates.

    Note: `trace` and `ema` have very similar but distinct updates;
    `ema = decay * ema + (1-decay) * t`, while `trace = decay * trace + t`.
    Both are frequently found in the optimisation literature.

    Args:
        decay: the decay rate for the exponential moving average.
        debias: whether to debias the transformed gradient.
        accumulator_dtype: optional `dtype` to used for the accumulator; if `None`
            then the `dtype` is inferred from `params` and `updates`.
    """
    decay: RealNumeric
    debias: bool = field(default=True, static=True)
    accumulator_dtype: Any = field(default=None, static=True)

    def init(self, parameters: Weights) -> EmaState:
        return ema(self.decay, self.debias, self.accumulator_dtype).init(parameters)

    def update(self,
               gradient: Weights,
               state: EmaState,
               parameters: Optional[Weights] = None) -> Tuple[Weights, EmaState]:
        return ema(self.decay, self.debias, self.accumulator_dtype).update(gradient, state,
                                                                           parameters)


@dataclass
class ScaleByRss(GradientTransformation[ScaleByRssState, Weights], Generic[Weights]):
    """Rescale updates by the root of the sum of all squared gradients to date.

    References:
        [Duchi et al, 2011](https://jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
        [McMahan et al., 2010](https://arxiv.org/abs/1002.4908)

    Args:
        initial_accumulator_value: Starting value for accumulators, must be >= 0.
        eps: A small floating point value to avoid zero denominator.
    """
    initial_accumulator_value: RealNumeric = 0.1
    eps: RealNumeric = 1e-7

    def init(self, parameters: Weights) -> ScaleByRssState:
        return scale_by_rss(self.initial_accumulator_value, self.eps).init(parameters)

    def update(self,
               gradient: Weights,
               state: ScaleByRssState,
               parameters: Optional[Weights] = None) -> Tuple[Weights, ScaleByRssState]:
        return scale_by_rss(self.initial_accumulator_value, self.eps).udpate(gradient, state,
                                                                             parameters)


@dataclass
class ScaleByRms(GradientTransformation[ScaleByRmsState, Weights], Generic[Weights]):
    """Rescale updates by the root of the exp. moving avg of the square.

    References:
        [Hinton](www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

    Args:
        decay: decay rate for the exponentially weighted average of squared grads.
        eps: term added to the denominator to improve numerical stability.
        initial_scale: initial value for second moment
    """
    decay: RealNumeric = 0.9
    eps: RealNumeric = 1e-8
    initial_scale: RealNumeric = 0.0

    def init(self, parameters: Weights) -> ScaleByRmsState:
        return scale_by_rms(self.decay, self.eps, self.initial_scale).init(parameters)

    def update(self,
               gradient: Weights,
               state: ScaleByRmsState,
               parameters: Optional[Weights]) -> Tuple[Weights, ScaleByRmsState]:
        return scale_by_rms(self.decay, self.eps, self.initial_scale).update(gradient, state,
                                                                             parameters)


@dataclass
class ScaleByStddev(GradientTransformation[ScaleByRStdDevState, Weights], Generic[Weights]):
    """Rescale updates by the root of the centered exp. moving average of squares.

    References:
        [Hinton](www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

    Args:
        decay: decay rate for the exponentially weighted average of squared grads.
        eps: term added to the denominator to improve numerical stability.
        initial_scale: initial value for second moment
    """
    decay: RealNumeric = 0.9
    eps: RealNumeric = 1e-8
    initial_scale: RealNumeric = 0.0

    def init(self, parameters: Weights) -> ScaleByRStdDevState:
        return scale_by_stddev(self.decay, self.eps, self.initial_scale).init(parameters)

    def update(self,
               gradient: Weights,
               state: ScaleByRStdDevState,
               parameters: Optional[Weights]) -> Tuple[Weights, ScaleByRStdDevState]:
        return scale_by_stddev(self.decay, self.eps, self.initial_scale).update(gradient, state,
                                                                                parameters)


@dataclass
class ScaleByAdam(GradientTransformation[ScaleByAdamState, Weights], Generic[Weights]):
    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.999
    eps: RealNumeric = 1e-8
    eps_root: RealNumeric = 0.0
    mu_dtype: Any = field(default=None, static=True)  # TODO: use when optax exposes it.

    def init(self, parameters: Weights) -> ScaleByAdamState:
        return scale_by_adam(self.b1, self.b2, self.eps, self.eps_root).init(
            parameters)

    def update(self,
               gradient: Weights,
               state: ScaleByAdamState,
               parameters: Optional[Weights]) -> Tuple[Weights, ScaleByAdamState]:
        return scale_by_adam(self.b1, self.b2, self.eps, self.eps_root).update(
            gradient, state, parameters)


@dataclass
class Scale(GradientTransformation[ScaleState, Weights], Generic[Weights]):
    """Scale updates by some fixed scalar `step_size`.

    Args:
        step_size: a scalar corresponding to a fixed scaling factor for updates.
    """
    step_size: RealNumeric

    def init(self, parameters: Weights) -> ScaleState:
        return scale(self.step_size).init(parameters)

    def update(self,
               gradient: Weights,
               state: ScaleState,
               parameters: Optional[Weights]) -> Tuple[Weights, ScaleState]:
        return scale(self.step_size).update(gradient, state, parameters)


@dataclass
class ScaleByParamBlockNorm(GradientTransformation[EmptyState, Weights], Generic[Weights]):
    """Scale updates for each param block by the norm of that block's parameters.

    A `block` is here a weight vector (e.g. in a Linear layer) or a weight matrix
    (e.g. in a convolutional layer) appearing as a leaf in the grads/param pytree.

    Args:
        min_scale: minimum scaling factor.
    """
    min_scale: RealNumeric = 1e-3

    def init(self, parameters: Weights) -> EmptyState:
        return scale_by_param_block_norm(self.min_scale).init(parameters)

    def update(self,
               gradient: Weights,
               state: EmptyState,
               parameters: Optional[Weights]) -> Tuple[Weights, EmptyState]:
        return scale_by_param_block_norm(self.min_scale).update(gradient, state, parameters)


@dataclass
class ScaleByParamBlockRMS(GradientTransformation[EmptyState, Weights], Generic[Weights]):
    """Scale updates by rms of the gradient for each param vector or matrix.

    A `block` is here a weight vector (e.g. in a Linear layer) or a weight matrix
    (e.g. in a convolutional layer) appearing as a leaf in the grads/param pytree.

    Args:
        min_scale: minimum scaling factor.
    """
    min_scale: RealNumeric = 1e-3

    def init(self, parameters: Weights) -> EmptyState:
        return scale_by_param_block_rms(self.min_scale).init(parameters)

    def update(self,
               gradient: Weights,
               state: EmptyState,
               parameters: Optional[Weights]) -> Tuple[Weights, EmptyState]:
        return scale_by_param_block_rms(self.min_scale).update(gradient, state, parameters)


@dataclass
class ScaleByBelief(GradientTransformation[ScaleByBeliefState, Weights], Generic[Weights]):
    """Rescale updates according to the AdaBelief algorithm.

    References:
      [Zhuang et al, 2020](https://arxiv.org/abs/2010.07468)

    Args:
      b1: decay rate for the exponentially weighted average of grads.
      b2: decay rate for the exponentially weighted average of variance of grads.
      eps: term added to the denominator to improve numerical stability.
      eps_root: term added to the denominator inside the square-root to improve
        numerical stability when backpropagating gradients through the rescaling.
    """
    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.999
    eps: RealNumeric = 0.0
    eps_root: RealNumeric = 1e-16

    def init(self, parameters: Weights) -> EmptyState:
        return scale_by_belief(self.b1, self.b2, self.eps, self.eps_root).init(parameters)

    def update(self,
               gradient: Weights,
               state: EmptyState,
               parameters: Optional[Weights]) -> Tuple[Weights, EmptyState]:
        return scale_by_belief(self.b1, self.b2, self.eps, self.eps_root).update(gradient, state,
                                                                                 parameters)


@dataclass
class ScaleByYogi(GradientTransformation[ScaleByAdamState, Weights], Generic[Weights]):
    """Rescale updates according to the Yogi algorithm.

    References:
        [Zaheer et al, 2018](https://papers.nips.cc/paper/2018/hash/90365351ccc7437a1309dc64e4db32a3-Abstract.html) #pylint:disable=line-too-long

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

    def init(self, parameters: Weights) -> EmptyState:
        return scale_by_yogi(self.b1, self.b2, self.eps, self.eps_root).init(parameters)

    def update(self,
               gradient: Weights,
               state: EmptyState,
               parameters: Optional[Weights]) -> Tuple[Weights, EmptyState]:
        return scale_by_yogi(self.b1, self.b2, self.eps, self.eps_root).update(gradient, state,
                                                                               parameters)


@dataclass
class ScaleByRAdam(GradientTransformation[ScaleByAdamState, Weights], Generic[Weights]):
    """Rescale updates according to the Rectified Adam algorithm.

    References:
        [Liu et al, 2020](https://arxiv.org/abs/1908.03265)

    Args:
        b1: decay rate for the exponentially weighted average of grads.
        b2: decay rate for the exponentially weighted average of squared grads.
        eps: term added to the denominator to improve numerical stability.
        eps_root: term added to the denominator inside the square-root to improve
            numerical stability when backpropagating gradients through the rescaling.
        threshold: Threshold for variance tractability
    """
    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.999
    eps: RealNumeric = 1e-8
    eps_root: RealNumeric = 0.0
    threshold: RealNumeric = 5.0

    def init(self, parameters: Weights) -> EmptyState:
        return scale_by_radam(self.b1, self.b2, self.eps, self.eps_root).init(parameters)

    def update(self,
               gradient: Weights,
               state: EmptyState,
               parameters: Optional[Weights]) -> Tuple[Weights, EmptyState]:
        return scale_by_radam(self.b1, self.b2, self.eps, self.eps_root).update(gradient, state,
                                                                               parameters)

@dataclass
class AddDecayedWeights(GradientTransformation[EmptyState, Weights], Generic[Weights]):
    """Add parameter scaled by `weight_decay`.

    Args:
        weight_decay: a scalar weight decay rate.
        mask: a tree with same structure as (or a prefix of) the params PyTree,
            or a Callable that returns such a pytree given the params/updates.
            The leaves should be booleans, `True` for leaves/subtrees you want to
            apply the transformation to, and `False` for those you want to skip.
        weight_decay: RealNumeric = 0.0
    """
    weight_decay: RealNumeric = 0.0
    mask: Optional[Union[Any, Callable[[Weights], Any]]] = field(default=None, static=True)

    def init(self, parameters: Weights) -> EmptyState:
        return add_decayed_weights(self.weight_decay, self.mask).init(parameters)

    def update(self,
               gradient: Weights,
               state: EmptyState,
               parameters: Optional[Weights]) -> Tuple[Weights, EmptyState]:
        return add_decayed_weights(self.weight_decay, self.mask).update(gradient, state, parameters)


@dataclass
class ScaleBySchedule(GradientTransformation[ScaleByScheduleState, Weights], Generic[Weights]):
    """Scale updates using a custom schedule for the `step_size`.

    Args:
        step_size_fn: a function that takes an update count as input and proposes
            the step_size to multiply the updates by.
    """
    step_size_fn: Schedule = field(static=True)

    def init(self, parameters: Weights) -> ScaleByScheduleState:
        return scale_by_schedule(self.step_size_fn).init(parameters)

    def update(self,
               gradient: Weights,
               state: ScaleByScheduleState,
               parameters: Optional[Weights]) -> Tuple[Weights, ScaleByScheduleState]:
        return scale_by_schedule(self.step_size_fn).update(gradient, state, parameters)


@dataclass
class ScaleByTrustRatio(GradientTransformation[ScaleByTrustRatioState, Weights], Generic[Weights]):
    """Scale updates by trust ratio`.

    References:
        [You et. al 2020](https://arxiv.org/abs/1904.00962)

    Args:
        min_norm: minimum norm for params and gradient norms; by default is zero.
        trust_coefficient: a multiplier for the trust ratio.
        eps: additive constant added to the denominator for numerical stability.
    """
    min_norm: RealNumeric = 0.0
    trust_coefficient: RealNumeric = 1.0
    eps: RealNumeric = 0.0

    def init(self, parameters: Weights) -> ScaleByTrustRatioState:
        return scale_by_trust_ratio(self.min_norm, self.trust_coefficient,
                                    self.eps).init(parameters)

    def update(self,
               gradient: Weights,
               state: ScaleByTrustRatioState,
               parameters: Optional[Weights]) -> Tuple[Weights, ScaleByTrustRatioState]:
        return scale_by_trust_ratio(self.min_norm, self.trust_coefficient,
                                    self.eps).update(gradient, state, parameters)


@dataclass
class AddNoise(GradientTransformation[AddNoiseState, Weights], Generic[Weights]):
    """Add gradient noise.

    References:
        [Neelakantan et al, 2014](https://arxiv.org/abs/1511.06807)

    Args:
        eta: base variance of the gaussian noise added to the gradient.
        gamma: decay exponent for annealing of the variance.
        seed: seed for random number generation.
    """
    eta: RealNumeric
    gamma: RealNumeric
    rng: Generator

    def init(self, parameters: Weights) -> AddNoiseState:
        return AddNoiseState(count=jnp.zeros([], jnp.int32), rng_key=self.rng.key)

    def update(self,
               gradient: Weights,
               state: AddNoiseState,
               parameters: Optional[Weights]) -> Tuple[Weights, AddNoiseState]:
        return add_noise(self.eta, self.gamma, 0).update(gradient, state, parameters)


@dataclass
class ApplyEvery(GradientTransformation[ApplyEveryState, Weights], Generic[Weights]):
    """Accumulate gradients and apply them every k steps.

    Note that if this transformation is part of a chain, the states of the other
    transformations will still be updated at every step. In particular, using
    `apply_every` with a batch size of N/2 and k=2 is not necessarily equivalent
    to not using `apply_every` with a batch size of N. If this equivalence is
    important for you, consider using the `optax.MultiSteps`.

    Args:
        k: emit non-zero gradients every k steps, otherwise accumulate them.
    """
    k: int = field(default=1, static=True)

    def init(self, parameters: Weights) -> ApplyEveryState:
        return apply_every(self.k).init(parameters)

    def update(self,
               gradient: Weights,
               state: ApplyEveryState,
               parameters: Optional[Weights]) -> Tuple[Weights, ApplyEveryState]:
        return apply_every(self.k).update(gradient, state, parameters)


@dataclass
class Centralize(GradientTransformation[EmptyState, Weights], Generic[Weights]):
    """Centralize gradients.

    References:
        [Yong et al, 2020](https://arxiv.org/abs/2004.01461)
    """
    def init(self, parameters: Weights) -> EmptyState:
        return centralize().init(parameters)

    def update(self,
               gradient: Weights,
               state: ApplyEveryState,
               parameters: Optional[Weights]) -> Tuple[Weights, ApplyEveryState]:
        return centralize().update(gradient, state, parameters)


@dataclass
class ScaleBySM3(GradientTransformation[ScaleBySM3State, Weights], Generic[Weights]):
    """Scale updates by sm3`.

    References:
        [Anil et. al 2019](https://arxiv.org/abs/1901.11150)

    Args:
        b1: decay rate for the exponentially weighted average of grads.
        b2: decay rate for the exponentially weighted average of squared grads.
        eps: term added to the denominator to improve numerical stability.
    """
    b1: RealNumeric = 0.9
    b2: RealNumeric = 1.0
    eps: RealNumeric = 1e-8

    def init(self, parameters: Weights) -> ScaleBySM3State:
        return scale_by_sm3(self.b1, self.b2, self.eps).init(parameters)

    def update(self,
               gradient: Weights,
               state: ScaleBySM3State,
               parameters: Optional[Weights]) -> Tuple[Weights, ScaleBySM3State]:
        return scale_by_sm3(self.b1, self.b2, self.eps).update(gradient, state, parameters)
