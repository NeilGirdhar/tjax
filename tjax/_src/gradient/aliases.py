from __future__ import annotations

from collections.abc import Callable
from dataclasses import KW_ONLY, asdict
from typing import Any, Generic

import jax.numpy as jnp
from optax import (GradientTransformationExtraArgs, adabelief, adadelta, adafactor, adagrad, adam,
                   adamax, adamaxw, adamw, fromage, lamb, lars, lbfgs, lion, noisy_sgd, novograd,
                   optimistic_gradient_descent, polyak_sgd, radam, rmsprop,
                   scale_by_zoom_linesearch, sgd, sm3, yogi)
from optax.contrib import dpsgd
from typing_extensions import override

from tjax.dataclasses import dataclass, field

from ..annotations import IntegralNumeric, JaxArray, RealNumeric
from .transform import GenericGradientState, GradientTransformation, Weights
from .transforms import Schedule

# Types --------------------------------------------------------------------------------------------
ScalarOrSchedule = float | JaxArray | Schedule


# Transforms from optax._src.alias.py --------------------------------------------------------------
@dataclass
class AdaBelief(GradientTransformation[GenericGradientState, Weights], Generic[Weights]):
    learning_rate: ScalarOrSchedule
    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.999
    eps: RealNumeric = 1e-8
    eps_root: RealNumeric = 1e-16

    @override
    def init(self, parameters: Weights
             ) -> GenericGradientState:
        return GenericGradientState(adabelief(**asdict(self)).init(parameters))

    @override
    def update(self,
               gradient: Weights,
               state: GenericGradientState,
               parameters: Weights | None
               ) -> tuple[Weights, GenericGradientState]:
        return GenericGradientState.wrap(  # pyright: ignore
            *adabelief(**asdict(self)).update(gradient, state.data, parameters))


@dataclass
class AdaDelta(GradientTransformation[GenericGradientState, Weights], Generic[Weights]):
    learning_rate: ScalarOrSchedule | None = None
    rho: RealNumeric = 0.9
    eps: RealNumeric = 1e-6
    weight_decay: RealNumeric = 0.0
    weight_decay_mask: bool | Weights | Callable[[Weights], Any] | None = field(default=None,
                                                                                static=True)

    @override
    def init(self, parameters: Weights
             ) -> GenericGradientState:
        return GenericGradientState(adadelta(**asdict(self)).init(parameters))

    @override
    def update(self,
               gradient: Weights,
               state: GenericGradientState,
               parameters: Weights | None
               ) -> tuple[Weights, GenericGradientState]:
        return GenericGradientState.wrap(  # pyright: ignore
            *adadelta(**asdict(self)).update(gradient, state.data, parameters))


@dataclass
class AdaFactor(GradientTransformation[GenericGradientState, Weights], Generic[Weights]):
    learning_rate: ScalarOrSchedule | None = None
    min_dim_size_to_factor: IntegralNumeric = 128
    decay_rate: RealNumeric = 0.8
    decay_offset: IntegralNumeric = 0
    multiply_by_parameter_scale: RealNumeric = True
    clipping_threshold: float | None = 1.0
    momentum: float | None = None
    dtype_momentum: Any = field(default=jnp.float32, static=True)
    weight_decay_rate: float | None = None
    eps: RealNumeric = 1e-30
    factored: bool = field(default=True, static=True)
    weight_decay_mask: bool | Weights | Callable[[Weights], Any] | None = field(default=None,
                                                                                static=True)

    @override
    def init(self, parameters: Weights
             ) -> GenericGradientState:
        return GenericGradientState(adafactor(**asdict(self)).init(parameters))

    @override
    def update(self,
               gradient: Weights,
               state: GenericGradientState,
               parameters: Weights | None
               ) -> tuple[Weights, GenericGradientState]:
        return GenericGradientState.wrap(  # pyright: ignore
            *adafactor(**asdict(self)).update(gradient, state.data, parameters))


@dataclass
class AdaGrad(GradientTransformation[GenericGradientState, Weights], Generic[Weights]):
    learning_rate: ScalarOrSchedule
    initial_accumulator_value: RealNumeric = 0.1
    eps: RealNumeric = 1e-7

    @override
    def init(self, parameters: Weights
             ) -> GenericGradientState:
        return GenericGradientState(adagrad(**asdict(self)).init(parameters))

    @override
    def update(self,
               gradient: Weights,
               state: GenericGradientState,
               parameters: Weights | None
               ) -> tuple[Weights, GenericGradientState]:
        return GenericGradientState.wrap(  # pyright: ignore
            *adagrad(**asdict(self)).update(gradient, state.data, parameters))


@dataclass
class Adam(GradientTransformation[GenericGradientState, Weights], Generic[Weights]):
    learning_rate: ScalarOrSchedule
    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.999
    eps: RealNumeric = 1e-8
    eps_root: RealNumeric = 0.0
    mu_dtype: Any | None = field(default=None, static=True)
    _: KW_ONLY
    nesterov: bool = field(default=False, static=True)

    @override
    def init(self, parameters: Weights
             ) -> GenericGradientState:
        return GenericGradientState(adam(**asdict(self)).init(parameters))

    @override
    def update(self,
               gradient: Weights,
               state: GenericGradientState,
               parameters: Weights | None
               ) -> tuple[Weights, GenericGradientState]:
        return GenericGradientState.wrap(  # pyright: ignore
            *adam(**asdict(self)).update(gradient, state.data, parameters))


@dataclass
class AdamW(GradientTransformation[GenericGradientState, Weights], Generic[Weights]):
    learning_rate: RealNumeric
    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.999
    eps: RealNumeric = 1e-8
    eps_root: RealNumeric = 0.0
    mu_dtype: Any | None = field(default=None, static=True)
    weight_decay: RealNumeric = 1e-4
    mask: Any | Callable[[Any], Any] | None = None
    _: KW_ONLY
    nesterov: bool = field(default=False, static=True)

    @override
    def init(self, parameters: Weights
             ) -> GenericGradientState:
        return GenericGradientState(adamw(**asdict(self)).init(parameters))

    @override
    def update(self,
               gradient: Weights,
               state: GenericGradientState,
               parameters: Weights | None
               ) -> tuple[Weights, GenericGradientState]:
        return GenericGradientState.wrap(  # pyright: ignore
            *adamw(**asdict(self)).update(gradient, state.data, parameters))


@dataclass
class Adamax(GradientTransformation[GenericGradientState, Weights], Generic[Weights]):
    learning_rate: RealNumeric
    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.999
    eps: RealNumeric = 1e-8

    @override
    def init(self, parameters: Weights
             ) -> GenericGradientState:
        return GenericGradientState(adamax(**asdict(self)).init(parameters))

    @override
    def update(self,
               gradient: Weights,
               state: GenericGradientState,
               parameters: Weights | None
               ) -> tuple[Weights, GenericGradientState]:
        return GenericGradientState.wrap(  # pyright: ignore
            *adamax(**asdict(self)).update(gradient, state.data, parameters))


@dataclass
class AdamaxW(GradientTransformation[GenericGradientState, Weights], Generic[Weights]):
    learning_rate: RealNumeric
    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.999
    eps: RealNumeric = 1e-8
    weight_decay: RealNumeric = 1e-4
    mask: Any | Callable[[Any], Any] | None = None

    @override
    def init(self, parameters: Weights
             ) -> GenericGradientState:
        return GenericGradientState(adamaxw(**asdict(self)).init(parameters))

    @override
    def update(self,
               gradient: Weights,
               state: GenericGradientState,
               parameters: Weights | None
               ) -> tuple[Weights, GenericGradientState]:
        return GenericGradientState.wrap(  # pyright: ignore
            *adamaxw(**asdict(self)).update(gradient, state.data, parameters))


@dataclass
class Fromage(GradientTransformation[GenericGradientState, Weights], Generic[Weights]):
    learning_rate: RealNumeric
    min_norm: RealNumeric = 1e-6

    @override
    def init(self, parameters: Weights
             ) -> GenericGradientState:
        return GenericGradientState(fromage(**asdict(self)).init(parameters))

    @override
    def update(self,
               gradient: Weights,
               state: GenericGradientState,
               parameters: Weights | None
               ) -> tuple[Weights, GenericGradientState]:
        return GenericGradientState.wrap(  # pyright: ignore
            *fromage(**asdict(self)).update(gradient, state.data, parameters))


@dataclass
class Lamb(GradientTransformation[GenericGradientState, Weights], Generic[Weights]):
    learning_rate: ScalarOrSchedule
    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.999
    eps: RealNumeric = 1e-6
    eps_root: RealNumeric = 0.0
    weight_decay: RealNumeric = 0.0
    mask: bool | Weights | Callable[[Weights], Any] | None = field(default=None, static=True)

    @override
    def init(self, parameters: Weights
             ) -> GenericGradientState:
        return GenericGradientState(lamb(**asdict(self)).init(parameters))

    @override
    def update(self,
               gradient: Weights,
               state: GenericGradientState,
               parameters: Weights | None
               ) -> tuple[Weights, GenericGradientState]:
        return GenericGradientState.wrap(  # pyright: ignore
            *lamb(**asdict(self)).update(gradient, state.data, parameters))


@dataclass
class LARS(GradientTransformation[GenericGradientState, Weights], Generic[Weights]):
    learning_rate: ScalarOrSchedule
    weight_decay: RealNumeric = 0.0
    weight_decay_mask: bool | Weights | Callable[[Weights], Any] | None = field(default=True,
                                                                                static=True)
    trust_coefficient: RealNumeric = 0.001
    eps: RealNumeric = 0.0
    trust_ratio_mask: bool | Weights | Callable[[Weights], Any] | None = field(default=True,
                                                                               static=True)
    momentum: RealNumeric = 0.9
    nesterov: bool = field(default=False, static=True)

    @override
    def init(self, parameters: Weights
             ) -> GenericGradientState:
        return GenericGradientState(lars(**asdict(self)).init(parameters))

    @override
    def update(self,
               gradient: Weights,
               state: GenericGradientState,
               parameters: Weights | None
               ) -> tuple[Weights, GenericGradientState]:
        return GenericGradientState.wrap(  # pyright: ignore
            *lars(**asdict(self)).update(gradient, state.data, parameters))


@dataclass
class LBFGS(GradientTransformation[GenericGradientState, Weights], Generic[Weights]):
    learning_rate: ScalarOrSchedule | None = None
    memory_size: int = field(default=10, static=True)
    scale_init_precond: bool = field(default=True, static=True)
    linesearch: GradientTransformationExtraArgs | None = (
            scale_by_zoom_linesearch(max_linesearch_steps=15))

    @override
    def init(self, parameters: Weights
             ) -> GenericGradientState:
        return GenericGradientState(lars(**asdict(self)).init(parameters))

    @override
    def update(self,
               gradient: Weights,
               state: GenericGradientState,
               parameters: Weights | None
               ) -> tuple[Weights, GenericGradientState]:
        # TODO: The update needs parameters.
        return GenericGradientState.wrap(  # pyright: ignore
            *lbfgs(**asdict(self)).update(gradient, state.data, parameters))


@dataclass
class Lion(GradientTransformation[GenericGradientState, Weights], Generic[Weights]):
    learning_rate: ScalarOrSchedule
    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.99
    mu_dtype: Any | None = field(default=None, static=True)
    weight_decay: RealNumeric = 1e-3
    mask: Any | Callable[[Any], Any] | None = None

    @override
    def init(self, parameters: Weights
             ) -> GenericGradientState:
        return GenericGradientState(lion(**asdict(self)).init(parameters))

    @override
    def update(self,
               gradient: Weights,
               state: GenericGradientState,
               parameters: Weights | None
               ) -> tuple[Weights, GenericGradientState]:
        return GenericGradientState.wrap(  # pyright: ignore
            *lion(**asdict(self)).update(gradient, state.data, parameters))


@dataclass
class NoisySGD(GradientTransformation[GenericGradientState, Weights], Generic[Weights]):
    learning_rate: ScalarOrSchedule
    eta: RealNumeric = 0.01
    gamma: RealNumeric = 0.55
    seed: IntegralNumeric = 0

    @override
    def init(self, parameters: Weights
             ) -> GenericGradientState:
        return GenericGradientState(noisy_sgd(**asdict(self)).init(parameters))

    @override
    def update(self,
               gradient: Weights,
               state: GenericGradientState,
               parameters: Weights | None
               ) -> tuple[Weights, GenericGradientState]:
        return GenericGradientState.wrap(  # pyright: ignore
            *noisy_sgd(**asdict(self)).update(gradient, state.data, parameters))


@dataclass
class Novograd(GradientTransformation[GenericGradientState, Weights], Generic[Weights]):
    learning_rate: ScalarOrSchedule
    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.25
    eps: RealNumeric = 1e-6
    eps_root: RealNumeric = 0.0
    weight_decay: RealNumeric = 0.0

    @override
    def init(self, parameters: Weights
             ) -> GenericGradientState:
        return GenericGradientState(novograd(**asdict(self)).init(parameters))

    @override
    def update(self,
               gradient: Weights,
               state: GenericGradientState,
               parameters: Weights | None
               ) -> tuple[Weights, GenericGradientState]:
        return GenericGradientState.wrap(  # pyright: ignore
            *novograd(**asdict(self)).update(gradient, state.data, parameters))


@dataclass
class OptimisticGradientDescent(GradientTransformation[GenericGradientState, Weights],
                                Generic[Weights]):
    learning_rate: ScalarOrSchedule
    alpha: ScalarOrSchedule
    beta: ScalarOrSchedule

    @override
    def init(self, parameters: Weights
             ) -> GenericGradientState:
        return GenericGradientState(optimistic_gradient_descent(**asdict(self)).init(parameters))

    @override
    def update(self,
               gradient: Weights,
               state: GenericGradientState,
               parameters: Weights | None
               ) -> tuple[Weights, GenericGradientState]:
        return GenericGradientState.wrap(  # pyright: ignore
            *optimistic_gradient_descent(**asdict(self)).update(gradient, state.data, parameters))


@dataclass
class PolyakSGD(GradientTransformation[GenericGradientState, Weights], Generic[Weights]):
    max_learning_rate: RealNumeric = 1.0
    scaling: ScalarOrSchedule = 1.0
    f_min: RealNumeric = 0.0
    eps: RealNumeric = 0.0

    @override
    def init(self, parameters: Weights
             ) -> GenericGradientState:
        return GenericGradientState(polyak_sgd(**asdict(self)).init(parameters))

    @override
    def update(self,
               gradient: Weights,
               state: GenericGradientState,
               parameters: Weights | None
               ) -> tuple[Weights, GenericGradientState]:
        # TODO: The update needs parameters.
        return GenericGradientState.wrap(  # pyright: ignore
            *polyak_sgd(**asdict(self)).update(gradient, state.data, parameters))


@dataclass
class RAdam(GradientTransformation[GenericGradientState, Weights], Generic[Weights]):
    learning_rate: ScalarOrSchedule
    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.999
    eps: RealNumeric = 1e-8
    eps_root: RealNumeric = 0.0
    threshold: RealNumeric = 5.0
    _: KW_ONLY
    nesterov: bool = field(default=False, static=True)

    @override
    def init(self, parameters: Weights
             ) -> GenericGradientState:
        return GenericGradientState(radam(**asdict(self)).init(parameters))

    @override
    def update(self,
               gradient: Weights,
               state: GenericGradientState,
               parameters: Weights | None
               ) -> tuple[Weights, GenericGradientState]:
        return GenericGradientState.wrap(  # pyright: ignore
            *radam(**asdict(self)).update(gradient, state.data, parameters))


@dataclass
class RMSProp(GradientTransformation[GenericGradientState, Weights], Generic[Weights]):
    """A flexible RMSProp optimiser.

    RMSProp is an SGD variant with learning rate adaptation. The `learning_rate`
    used for each weight is scaled by a suitable estimate of the magnitude of the
    gradients on previous steps. Several variants of RMSProp can be found
    in the literature. This alias provides an easy to configure RMSProp
    optimiser that can be used to switch between several of these variants.

    References:
        Tieleman and Hinton, 2012: http://www.cs.toronto.edu/~hinton/coursera/lecture6/lec6.pdf
        Graves, 2013: https://arxiv.org/abs/1308.0850
    Args:
        learning_rate: this is a fixed global scaling factor.
        decay: the decay used to track the magnitude of previous gradients.
        eps: a small numerical constant to avoid dividing by zero when rescaling.
        initial_scale: (default `0.`), initialisation of accumulators tracking the
        magnitude of previous updates. PyTorch uses `0`, TF1 uses `1`. When
        reproducing results from a paper, verify the value used by the authors.
        centered: (default `False`), whether the second moment or the variance of
        the past gradients is used to rescale the latest gradients.
        momentum: (default `None`), the `decay` rate used by the momentum term,
        when it is set to `None`, then momentum is not used at all.
        nesterov (default `False`): whether nesterov momentum is used.
    """
    learning_rate: RealNumeric
    decay: RealNumeric = 0.9
    eps: RealNumeric = 1e-8
    initial_scale: RealNumeric = 0.0
    centered: bool = field(default=False, static=True)
    momentum: float | None = None
    nesterov: bool = field(default=False, static=True)
    bias_correction: bool = field(default=False, static=True)

    @override
    def init(self, parameters: Weights
             ) -> GenericGradientState:
        return GenericGradientState(rmsprop(**asdict(self)).init(parameters))

    @override
    def update(self,
               gradient: Weights,
               state: GenericGradientState,
               parameters: Weights | None
               ) -> tuple[Weights, GenericGradientState]:
        return GenericGradientState.wrap(  # pyright: ignore
            *rmsprop(**asdict(self)).update(gradient, state.data, parameters))


@dataclass
class SGD(GradientTransformation[GenericGradientState, Weights], Generic[Weights]):
    """A canonical Stochastic Gradient Descent optimiser.

    This implements stochastic gradient descent. It also includes support for
    momentum, and nesterov acceleration, as these are standard practice when
    using stochastic gradient descent to train deep neural networks.

    References:
        Sutskever et al, 2013: http://proceedings.mlr.press/v28/sutskever13.pdf

    Args:
        learning_rate: this is a fixed global scaling factor.
        momentum: (default `None`), the `decay` rate used by the momentum term,
        when it is set to `None`, then momentum is not used at all.
        nesterov (default `False`): whether nesterov momentum is used.
        accumulator_dtype: optional `dtype` to be used for the accumulator; if
        `None` then the `dtype` is inferred from `params` and `updates`.

    Returns:
        A `GradientTransformation`.
    """
    learning_rate: RealNumeric
    momentum: float | None = None
    nesterov: bool = field(default=False, static=True)
    accumulator_dtype: Any | None = field(default=None, static=True)

    @override
    def init(self, parameters: Weights
             ) -> GenericGradientState:
        return GenericGradientState(sgd(**asdict(self)).init(parameters))

    @override
    def update(self,
               gradient: Weights,
               state: GenericGradientState,
               parameters: Weights | None
               ) -> tuple[Weights, GenericGradientState]:
        return GenericGradientState.wrap(  # pyright: ignore
            *sgd(**asdict(self)).update(gradient, state.data, parameters))


@dataclass
class SM3(GradientTransformation[GenericGradientState, Weights], Generic[Weights]):
    learning_rate: RealNumeric
    momentum: RealNumeric = 0.9

    @override
    def init(self, parameters: Weights
             ) -> GenericGradientState:
        return GenericGradientState(sm3(**asdict(self)).init(parameters))

    @override
    def update(self,
               gradient: Weights,
               state: GenericGradientState,
               parameters: Weights | None
               ) -> tuple[Weights, GenericGradientState]:
        return GenericGradientState.wrap(  # pyright: ignore
            *sm3(**asdict(self)).update(gradient, state.data, parameters))


@dataclass
class Yogi(GradientTransformation[GenericGradientState, Weights], Generic[Weights]):
    learning_rate: ScalarOrSchedule
    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.999
    eps: RealNumeric = 1e-3

    @override
    def init(self, parameters: Weights
             ) -> GenericGradientState:
        return GenericGradientState(yogi(**asdict(self)).init(parameters))

    @override
    def update(self,
               gradient: Weights,
               state: GenericGradientState,
               parameters: Weights | None
               ) -> tuple[Weights, GenericGradientState]:
        return GenericGradientState.wrap(  # pyright: ignore
            *yogi(**asdict(self)).update(gradient, state.data, parameters))


@dataclass
class DPSGD(GradientTransformation[GenericGradientState, Weights], Generic[Weights]):
    learning_rate: ScalarOrSchedule
    l2_norm_clip: RealNumeric
    noise_multiplier: RealNumeric
    seed: IntegralNumeric
    momentum: float | None = None
    nesterov: bool = field(default=False, static=True)

    @override
    def init(self, parameters: Weights
             ) -> GenericGradientState:
        return GenericGradientState(dpsgd(**asdict(self)).init(parameters))

    @override
    def update(self,
               gradient: Weights,
               state: GenericGradientState,
               parameters: Weights | None
               ) -> tuple[Weights, GenericGradientState]:
        return GenericGradientState.wrap(  # pyright: ignore
            *dpsgd(**asdict(self)).update(gradient, state.data, parameters))
