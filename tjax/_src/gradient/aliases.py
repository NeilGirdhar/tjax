from __future__ import annotations

from collections.abc import Callable
from dataclasses import KW_ONLY
from typing import Any, override

import jax.numpy as jnp
from optax import (
    GradientTransformationExtraArgs,
    adabelief,
    adadelta,
    adafactor,
    adagrad,
    adam,
    adamax,
    adamaxw,
    adamw,
    fromage,
    lamb,
    lars,
    lbfgs,
    lion,
    noisy_sgd,
    novograd,
    optimistic_gradient_descent,
    polyak_sgd,
    radam,
    rmsprop,
    scale_by_zoom_linesearch,
    sgd,
    sm3,
    yogi,
)
from optax.contrib import dpsgd

from tjax._src.annotations import IntegralNumeric, JaxArray, KeyArray, PyTree, RealNumeric
from tjax.dataclasses import as_shallow_dict, dataclass, field

from .transform import GenericGradientState, GradientTransformation
from .transforms import Schedule

# Types --------------------------------------------------------------------------------------------
ScalarOrSchedule = float | JaxArray | Schedule


# Transforms from optax._src.alias.py --------------------------------------------------------------
@dataclass
class AdaBelief[Weights: PyTree](GradientTransformation[GenericGradientState, Weights]):
    """Pytree-compatible wrapper for the optax AdaBelief optimiser.

    References:
        [Zhuang et al, 2020](https://arxiv.org/abs/2010.07468)

    Args:
        learning_rate: Global step-size schedule or scalar.
        b1: Decay rate for the first moment (mean) of gradients.
        b2: Decay rate for the second moment (variance) of gradients.
        eps: Regularisation term for numerical stability.
        eps_root: Term added inside the square-root for numerical stability.
    """

    learning_rate: ScalarOrSchedule
    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.999
    eps: RealNumeric = 1e-8
    eps_root: RealNumeric = 1e-16

    @override
    def init(self, parameters: Weights) -> GenericGradientState:
        return GenericGradientState(adabelief(**as_shallow_dict(self)).init(parameters))

    @override
    def update(
        self, gradient: Weights, state: GenericGradientState, parameters: Weights | None
    ) -> tuple[Weights, GenericGradientState]:
        return GenericGradientState.wrap(  # pyright: ignore # type: ignore
            *adabelief(**as_shallow_dict(self)).update(gradient, state.data, parameters)
        )


@dataclass
class AdaDelta[Weights: PyTree](GradientTransformation[GenericGradientState, Weights]):
    """Pytree-compatible wrapper for the optax AdaDelta optimiser.

    References:
        [Zeiler, 2012](https://arxiv.org/abs/1212.5701)

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
    weight_decay: RealNumeric = 0.0
    weight_decay_mask: bool | Weights | Callable[[Weights], Any] | None = field(
        default=None, static=True
    )

    @override
    def init(self, parameters: Weights) -> GenericGradientState:
        return GenericGradientState(adadelta(**as_shallow_dict(self)).init(parameters))

    @override
    def update(
        self, gradient: Weights, state: GenericGradientState, parameters: Weights | None
    ) -> tuple[Weights, GenericGradientState]:
        return GenericGradientState.wrap(  # pyright: ignore # type: ignore
            *adadelta(**as_shallow_dict(self)).update(gradient, state.data, parameters)
        )


@dataclass
class AdaFactor[Weights: PyTree](GradientTransformation[GenericGradientState, Weights]):
    """Pytree-compatible wrapper for the optax AdaFactor optimiser.

    AdaFactor reduces memory usage by factoring the second-moment estimate.

    References:
        [Shazeer and Stern, 2018](https://arxiv.org/abs/1805.09843)
    """

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
    weight_decay_mask: bool | Weights | Callable[[Weights], Any] | None = field(
        default=None, static=True
    )

    @override
    def init(self, parameters: Weights) -> GenericGradientState:
        return GenericGradientState(adafactor(**as_shallow_dict(self)).init(parameters))

    @override
    def update(
        self, gradient: Weights, state: GenericGradientState, parameters: Weights | None
    ) -> tuple[Weights, GenericGradientState]:
        return GenericGradientState.wrap(  # pyright: ignore # type: ignore
            *adafactor(**as_shallow_dict(self)).update(gradient, state.data, parameters)
        )


@dataclass
class AdaGrad[Weights: PyTree](GradientTransformation[GenericGradientState, Weights]):
    """Pytree-compatible wrapper for the optax AdaGrad optimiser.

    References:
        [Duchi et al, 2011](https://jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)

    Args:
        learning_rate: Global step-size schedule or scalar.
        initial_accumulator_value: Starting value for the sum-of-squares accumulator.
        eps: Regularisation term added to the denominator.
    """

    learning_rate: ScalarOrSchedule
    initial_accumulator_value: RealNumeric = 0.1
    eps: RealNumeric = 1e-7

    @override
    def init(self, parameters: Weights) -> GenericGradientState:
        return GenericGradientState(adagrad(**as_shallow_dict(self)).init(parameters))

    @override
    def update(
        self, gradient: Weights, state: GenericGradientState, parameters: Weights | None
    ) -> tuple[Weights, GenericGradientState]:
        return GenericGradientState.wrap(  # pyright: ignore # type: ignore
            *adagrad(**as_shallow_dict(self)).update(gradient, state.data, parameters)
        )


@dataclass
class Adam[Weights: PyTree](GradientTransformation[GenericGradientState, Weights]):
    """Pytree-compatible wrapper for the optax Adam optimiser.

    References:
        [Kingma and Ba, 2015](https://arxiv.org/abs/1412.6980)

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
    def init(self, parameters: Weights) -> GenericGradientState:
        return GenericGradientState(adam(**as_shallow_dict(self)).init(parameters))

    @override
    def update(
        self, gradient: Weights, state: GenericGradientState, parameters: Weights | None
    ) -> tuple[Weights, GenericGradientState]:
        return GenericGradientState.wrap(  # pyright: ignore # type: ignore
            *adam(**as_shallow_dict(self)).update(gradient, state.data, parameters)
        )


@dataclass
class AdamW[Weights: PyTree](GradientTransformation[GenericGradientState, Weights]):
    """Pytree-compatible wrapper for the optax AdamW optimiser.

    AdamW decouples the weight decay from the gradient-based update.

    References:
        [Loshchilov and Hutter, 2019](https://arxiv.org/abs/1711.05101)

    Args:
        learning_rate: Global step-size scalar.
        b1: Decay rate for the first moment of gradients.
        b2: Decay rate for the second moment of gradients.
        eps: Regularisation term for numerical stability.
        eps_root: Term added inside the square-root for numerical stability.
        mu_dtype: Optional dtype for the first moment accumulator.
        weight_decay: Coefficient for decoupled weight decay.
        mask: Optional mask for which parameters receive weight decay.
        nesterov: Whether to use Nesterov momentum.
    """

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
    def init(self, parameters: Weights) -> GenericGradientState:
        return GenericGradientState(adamw(**as_shallow_dict(self)).init(parameters))

    @override
    def update(
        self, gradient: Weights, state: GenericGradientState, parameters: Weights | None
    ) -> tuple[Weights, GenericGradientState]:
        return GenericGradientState.wrap(  # pyright: ignore # type: ignore
            *adamw(**as_shallow_dict(self)).update(gradient, state.data, parameters)
        )


@dataclass
class Adamax[Weights: PyTree](GradientTransformation[GenericGradientState, Weights]):
    """Pytree-compatible wrapper for the optax Adamax optimiser.

    Adamax is a variant of Adam based on the infinity norm.

    References:
        [Kingma and Ba, 2015](https://arxiv.org/abs/1412.6980)

    Args:
        learning_rate: Global step-size scalar.
        b1: Decay rate for the first moment of gradients.
        b2: Decay rate for the infinity norm of gradients.
        eps: Regularisation term for numerical stability.
    """

    learning_rate: RealNumeric
    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.999
    eps: RealNumeric = 1e-8

    @override
    def init(self, parameters: Weights) -> GenericGradientState:
        return GenericGradientState(adamax(**as_shallow_dict(self)).init(parameters))

    @override
    def update(
        self, gradient: Weights, state: GenericGradientState, parameters: Weights | None
    ) -> tuple[Weights, GenericGradientState]:
        return GenericGradientState.wrap(  # pyright: ignore # type: ignore
            *adamax(**as_shallow_dict(self)).update(gradient, state.data, parameters)
        )


@dataclass
class AdamaxW[Weights: PyTree](GradientTransformation[GenericGradientState, Weights]):
    """Pytree-compatible wrapper for the optax AdamaxW optimiser.

    AdamaxW combines the infinity-norm update of Adamax with decoupled weight
    decay from AdamW.

    Args:
        learning_rate: Global step-size scalar.
        b1: Decay rate for the first moment of gradients.
        b2: Decay rate for the infinity norm of gradients.
        eps: Regularisation term for numerical stability.
        weight_decay: Coefficient for decoupled weight decay.
        mask: Optional mask for which parameters receive weight decay.
    """

    learning_rate: RealNumeric
    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.999
    eps: RealNumeric = 1e-8
    weight_decay: RealNumeric = 1e-4
    mask: Any | Callable[[Any], Any] | None = None

    @override
    def init(self, parameters: Weights) -> GenericGradientState:
        return GenericGradientState(adamaxw(**as_shallow_dict(self)).init(parameters))

    @override
    def update(
        self, gradient: Weights, state: GenericGradientState, parameters: Weights | None
    ) -> tuple[Weights, GenericGradientState]:
        return GenericGradientState.wrap(  # pyright: ignore # type: ignore
            *adamaxw(**as_shallow_dict(self)).update(gradient, state.data, parameters)
        )


@dataclass
class Fromage[Weights: PyTree](GradientTransformation[GenericGradientState, Weights]):
    """Pytree-compatible wrapper for the optax Fromage optimiser.

    Fromage keeps parameter updates within a trust region defined by the
    relative distance in function space.

    References:
        [Bernstein et al, 2020](https://arxiv.org/abs/2002.03432)

    Args:
        learning_rate: Global step-size scalar.
        min_norm: Minimum norm used in the trust-region constraint.
    """

    learning_rate: RealNumeric
    min_norm: RealNumeric = 1e-6

    @override
    def init(self, parameters: Weights) -> GenericGradientState:
        return GenericGradientState(fromage(**as_shallow_dict(self)).init(parameters))

    @override
    def update(
        self, gradient: Weights, state: GenericGradientState, parameters: Weights | None
    ) -> tuple[Weights, GenericGradientState]:
        return GenericGradientState.wrap(  # pyright: ignore # type: ignore
            *fromage(**as_shallow_dict(self)).update(gradient, state.data, parameters)
        )


@dataclass
class Lamb[Weights: PyTree](GradientTransformation[GenericGradientState, Weights]):
    """Pytree-compatible wrapper for the optax LAMB optimiser.

    LAMB scales Adam updates by the ratio of the parameter norm to the update
    norm, which enables large-batch training.

    References:
        [You et al, 2020](https://arxiv.org/abs/1904.00962)

    Args:
        learning_rate: Global step-size schedule or scalar.
        b1: Decay rate for the first moment of gradients.
        b2: Decay rate for the second moment of gradients.
        eps: Regularisation term for numerical stability.
        eps_root: Term added inside the square-root for numerical stability.
        weight_decay: Coefficient for L2 regularisation.
        mask: Optional mask for which parameters receive weight decay.
    """

    learning_rate: ScalarOrSchedule
    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.999
    eps: RealNumeric = 1e-6
    eps_root: RealNumeric = 0.0
    weight_decay: RealNumeric = 0.0
    mask: bool | Weights | Callable[[Weights], Any] | None = field(default=None, static=True)

    @override
    def init(self, parameters: Weights) -> GenericGradientState:
        return GenericGradientState(lamb(**as_shallow_dict(self)).init(parameters))

    @override
    def update(
        self, gradient: Weights, state: GenericGradientState, parameters: Weights | None
    ) -> tuple[Weights, GenericGradientState]:
        return GenericGradientState.wrap(  # pyright: ignore # type: ignore
            *lamb(**as_shallow_dict(self)).update(gradient, state.data, parameters)
        )


@dataclass
class LARS[Weights: PyTree](GradientTransformation[GenericGradientState, Weights]):
    """Pytree-compatible wrapper for the optax LARS optimiser.

    Layer-wise Adaptive Rate Scaling adapts the learning rate per layer based
    on the ratio of the parameter norm to the gradient norm.

    References:
        [You et al, 2017](https://arxiv.org/abs/1708.03888)

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
    weight_decay: RealNumeric = 0.0
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
    def init(self, parameters: Weights) -> GenericGradientState:
        return GenericGradientState(lars(**as_shallow_dict(self)).init(parameters))

    @override
    def update(
        self, gradient: Weights, state: GenericGradientState, parameters: Weights | None
    ) -> tuple[Weights, GenericGradientState]:
        return GenericGradientState.wrap(  # pyright: ignore # type: ignore
            *lars(**as_shallow_dict(self)).update(gradient, state.data, parameters)
        )


@dataclass
class LBFGS[Weights: PyTree](GradientTransformation[GenericGradientState, Weights]):
    """Pytree-compatible wrapper for the optax L-BFGS optimiser.

    Limited-memory BFGS approximates the inverse Hessian using a fixed-size
    history of gradient and parameter differences.

    Args:
        learning_rate: Optional step-size schedule or scalar.  When ``None``
            the linesearch determines the step size.
        memory_size: Number of past gradient/step pairs to retain.
        scale_init_precond: Whether to scale the initial preconditioner.
        linesearch: Optional linesearch transformation; defaults to zoom
            linesearch with 15 maximum steps.
    """

    learning_rate: ScalarOrSchedule | None = None
    memory_size: int = field(default=10, static=True)
    scale_init_precond: bool = field(default=True, static=True)
    linesearch: GradientTransformationExtraArgs | None = scale_by_zoom_linesearch(
        max_linesearch_steps=15
    )

    @override
    def init(self, parameters: Weights) -> GenericGradientState:
        return GenericGradientState(lbfgs(**as_shallow_dict(self)).init(parameters))

    @override
    def update(
        self, gradient: Weights, state: GenericGradientState, parameters: Weights | None
    ) -> tuple[Weights, GenericGradientState]:
        # TODO: The update needs parameters.
        return GenericGradientState.wrap(  # pyright: ignore # type: ignore
            *lbfgs(**as_shallow_dict(self)).update(gradient, state.data, parameters)
        )


@dataclass
class Lion[Weights: PyTree](GradientTransformation[GenericGradientState, Weights]):
    """Pytree-compatible wrapper for the optax Lion optimiser.

    Lion (Evolved Sign Momentum) uses the sign of the update rather than the
    magnitude, reducing memory by eliminating the second-moment accumulator.

    References:
        [Chen et al, 2023](https://arxiv.org/abs/2302.06675)

    Args:
        learning_rate: Global step-size schedule or scalar.
        b1: Decay rate for the update momentum.
        b2: Decay rate for the second moment used to compute the update sign.
        mu_dtype: Optional dtype for the momentum accumulator.
        weight_decay: Coefficient for decoupled weight decay.
        mask: Optional mask for which parameters receive weight decay.
    """

    learning_rate: ScalarOrSchedule
    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.99
    mu_dtype: Any | None = field(default=None, static=True)
    weight_decay: RealNumeric = 1e-3
    mask: Any | Callable[[Any], Any] | None = None

    @override
    def init(self, parameters: Weights) -> GenericGradientState:
        return GenericGradientState(lion(**as_shallow_dict(self)).init(parameters))

    @override
    def update(
        self, gradient: Weights, state: GenericGradientState, parameters: Weights | None
    ) -> tuple[Weights, GenericGradientState]:
        return GenericGradientState.wrap(  # pyright: ignore # type: ignore
            *lion(**as_shallow_dict(self)).update(gradient, state.data, parameters)
        )


@dataclass
class NoisySGD[Weights: PyTree](GradientTransformation[GenericGradientState, Weights]):
    """Pytree-compatible wrapper for the optax Noisy SGD optimiser.

    Adds annealed Gaussian noise to the gradient at each step.

    References:
        [Neelakantan et al, 2015](https://arxiv.org/abs/1511.06807)

    Args:
        learning_rate: Global step-size schedule or scalar.
        eta: Base variance of the Gaussian noise.
        gamma: Decay exponent for the noise variance.
        seed: Seed for random noise generation.
    """

    learning_rate: ScalarOrSchedule
    key: KeyArray
    eta: RealNumeric = 0.01
    gamma: RealNumeric = 0.55

    @override
    def init(self, parameters: Weights) -> GenericGradientState:
        return GenericGradientState(noisy_sgd(**as_shallow_dict(self)).init(parameters))

    @override
    def update(
        self, gradient: Weights, state: GenericGradientState, parameters: Weights | None
    ) -> tuple[Weights, GenericGradientState]:
        return GenericGradientState.wrap(  # pyright: ignore # type: ignore
            *noisy_sgd(**as_shallow_dict(self)).update(gradient, state.data, parameters)
        )


@dataclass
class Novograd[Weights: PyTree](GradientTransformation[GenericGradientState, Weights]):
    """Pytree-compatible wrapper for the optax NovoGrad optimiser.

    NovoGrad normalises gradients layer-wise using the second moment,
    and uses a first moment similar to Adam.

    References:
        [Ginsburg et al, 2019](https://arxiv.org/abs/1905.11286)

    Args:
        learning_rate: Global step-size schedule or scalar.
        b1: Decay rate for the first moment of gradients.
        b2: Decay rate for the second moment of gradients.
        eps: Regularisation term for numerical stability.
        eps_root: Term added inside the square-root for numerical stability.
        weight_decay: Coefficient for L2 regularisation.
    """

    learning_rate: ScalarOrSchedule
    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.25
    eps: RealNumeric = 1e-6
    eps_root: RealNumeric = 0.0
    weight_decay: RealNumeric = 0.0

    @override
    def init(self, parameters: Weights) -> GenericGradientState:
        return GenericGradientState(novograd(**as_shallow_dict(self)).init(parameters))

    @override
    def update(
        self, gradient: Weights, state: GenericGradientState, parameters: Weights | None
    ) -> tuple[Weights, GenericGradientState]:
        return GenericGradientState.wrap(  # pyright: ignore # type: ignore
            *novograd(**as_shallow_dict(self)).update(gradient, state.data, parameters)
        )


@dataclass
class OptimisticGradientDescent[Weights: PyTree](
    GradientTransformation[GenericGradientState, Weights]
):
    """Pytree-compatible wrapper for the optax Optimistic Gradient Descent optimiser.

    Optimistic GD uses a prediction step to reduce variance and improve
    convergence in adversarial settings.

    References:
        [Daskalakis et al, 2017](https://arxiv.org/abs/1711.00141)

    Args:
        learning_rate: Global step-size schedule or scalar.
        alpha: Step-size for the gradient descent update.
        beta: Step-size for the optimistic gradient correction.
    """

    learning_rate: ScalarOrSchedule
    alpha: ScalarOrSchedule
    beta: ScalarOrSchedule

    @override
    def init(self, parameters: Weights) -> GenericGradientState:
        return GenericGradientState(
            optimistic_gradient_descent(**as_shallow_dict(self)).init(parameters)
        )

    @override
    def update(
        self, gradient: Weights, state: GenericGradientState, parameters: Weights | None
    ) -> tuple[Weights, GenericGradientState]:
        return GenericGradientState.wrap(  # pyright: ignore # type: ignore
            *optimistic_gradient_descent(**as_shallow_dict(self)).update(
                gradient, state.data, parameters
            )
        )


@dataclass
class PolyakSGD[Weights: PyTree](GradientTransformation[GenericGradientState, Weights]):
    """Pytree-compatible wrapper for the optax Polyak SGD optimiser.

    Uses a step size derived from the Polyak step rule, which aims to reach
    the minimum in one step under a quadratic loss assumption.

    References:
        [Polyak, 1987](https://doi.org/10.1016/C2013-0-03970-5)

    Args:
        max_learning_rate: Upper bound on the computed step size.
        scaling: Optional schedule or scalar to scale the Polyak step.
        f_min: Known or estimated minimum value of the loss.
        eps: Regularisation term added to the gradient norm denominator.
    """

    max_learning_rate: RealNumeric = 1.0
    scaling: ScalarOrSchedule = 1.0
    f_min: RealNumeric = 0.0
    eps: RealNumeric = 0.0

    @override
    def init(self, parameters: Weights) -> GenericGradientState:
        return GenericGradientState(polyak_sgd(**as_shallow_dict(self)).init(parameters))

    @override
    def update(
        self, gradient: Weights, state: GenericGradientState, parameters: Weights | None
    ) -> tuple[Weights, GenericGradientState]:
        # TODO: The update needs parameters.
        return GenericGradientState.wrap(  # pyright: ignore # type: ignore
            *polyak_sgd(**as_shallow_dict(self)).update(gradient, state.data, parameters)
        )


@dataclass
class RAdam[Weights: PyTree](GradientTransformation[GenericGradientState, Weights]):
    """Pytree-compatible wrapper for the optax RAdam optimiser.

    Rectified Adam stabilises Adam's warm-up phase by rectifying the variance
    of the adaptive learning rate.

    References:
        [Liu et al, 2020](https://arxiv.org/abs/1908.03265)

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
    def init(self, parameters: Weights) -> GenericGradientState:
        return GenericGradientState(radam(**as_shallow_dict(self)).init(parameters))

    @override
    def update(
        self, gradient: Weights, state: GenericGradientState, parameters: Weights | None
    ) -> tuple[Weights, GenericGradientState]:
        return GenericGradientState.wrap(  # pyright: ignore # type: ignore
            *radam(**as_shallow_dict(self)).update(gradient, state.data, parameters)
        )


@dataclass
class RMSProp[Weights: PyTree](GradientTransformation[GenericGradientState, Weights]):
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
    def init(self, parameters: Weights) -> GenericGradientState:
        return GenericGradientState(rmsprop(**as_shallow_dict(self)).init(parameters))

    @override
    def update(
        self, gradient: Weights, state: GenericGradientState, parameters: Weights | None
    ) -> tuple[Weights, GenericGradientState]:
        return GenericGradientState.wrap(  # pyright: ignore # type: ignore
            *rmsprop(**as_shallow_dict(self)).update(gradient, state.data, parameters)
        )


@dataclass
class SGD[Weights: PyTree](GradientTransformation[GenericGradientState, Weights]):
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
    def init(self, parameters: Weights) -> GenericGradientState:
        return GenericGradientState(sgd(**as_shallow_dict(self)).init(parameters))

    @override
    def update(
        self, gradient: Weights, state: GenericGradientState, parameters: Weights | None
    ) -> tuple[Weights, GenericGradientState]:
        return GenericGradientState.wrap(  # pyright: ignore # type: ignore
            *sgd(**as_shallow_dict(self)).update(gradient, state.data, parameters)
        )


@dataclass
class SM3[Weights: PyTree](GradientTransformation[GenericGradientState, Weights]):
    """Pytree-compatible wrapper for the optax SM3 optimiser.

    SM3 reduces memory by sharing accumulators across dimensions, making it
    suitable for very large parameter matrices.

    References:
        [Rohan and McMahan, 2019](https://arxiv.org/abs/1901.11150)

    Args:
        learning_rate: Global step-size scalar.
        momentum: Momentum decay rate.
    """

    learning_rate: RealNumeric
    momentum: RealNumeric = 0.9

    @override
    def init(self, parameters: Weights) -> GenericGradientState:
        return GenericGradientState(sm3(**as_shallow_dict(self)).init(parameters))

    @override
    def update(
        self, gradient: Weights, state: GenericGradientState, parameters: Weights | None
    ) -> tuple[Weights, GenericGradientState]:
        return GenericGradientState.wrap(  # pyright: ignore # type: ignore
            *sm3(**as_shallow_dict(self)).update(gradient, state.data, parameters)
        )


@dataclass
class Yogi[Weights: PyTree](GradientTransformation[GenericGradientState, Weights]):
    """Pytree-compatible wrapper for the optax Yogi optimiser.

    Yogi replaces Adam's additive second-moment update with a sign-based
    update to prevent the accumulator from growing too fast.

    References:
        [Zaheer et al, 2018](https://papers.nips.cc/paper/2018/hash/90365351ccc7437a1309dc64e4db32a3-Abstract.html)

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
    def init(self, parameters: Weights) -> GenericGradientState:
        return GenericGradientState(yogi(**as_shallow_dict(self)).init(parameters))

    @override
    def update(
        self, gradient: Weights, state: GenericGradientState, parameters: Weights | None
    ) -> tuple[Weights, GenericGradientState]:
        return GenericGradientState.wrap(  # pyright: ignore # type: ignore
            *yogi(**as_shallow_dict(self)).update(gradient, state.data, parameters)
        )


@dataclass
class DPSGD[Weights: PyTree](GradientTransformation[GenericGradientState, Weights]):
    """Pytree-compatible wrapper for the optax DP-SGD optimiser.

    Differentially Private SGD clips per-example gradients and adds Gaussian
    noise to provide (ε, δ)-differential privacy guarantees.

    References:
        [Abadi et al, 2016](https://arxiv.org/abs/1607.00133)

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
    def init(self, parameters: Weights) -> GenericGradientState:
        return GenericGradientState(dpsgd(**as_shallow_dict(self)).init(parameters))

    @override
    def update(
        self, gradient: Weights, state: GenericGradientState, parameters: Weights | None
    ) -> tuple[Weights, GenericGradientState]:
        return GenericGradientState.wrap(  # pyright: ignore # type: ignore
            *dpsgd(**as_shallow_dict(self)).update(gradient, state.data, parameters)
        )
