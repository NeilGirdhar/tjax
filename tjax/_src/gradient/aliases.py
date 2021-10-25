from typing import Any, Callable, Optional, Union

from .chain import ChainedGradientTransformation
from .transform import GradientTransformation
from .transforms import AddDecayedWeights, Scale, ScaleByAdam, Trace

__all__ = ['adam', 'adamw', 'sgd']


def adam(learning_rate: float,
         b1: float = 0.9,
         b2: float = 0.999,
         eps: float = 1e-8) -> GradientTransformation[Any, Any]:
    return ChainedGradientTransformation([ScaleByAdam(b1=b1, b2=b2, eps=eps),
                                          Scale(-learning_rate)])


def adamw(learning_rate: float,
          b1: float = 0.9,
          b2: float = 0.999,
          eps: float = 1e-8,
          eps_root: float = 0.0,
          mu_dtype: Optional[Any] = None,
          weight_decay: float = 1e-4,
          mask: Optional[Union[Any, Callable[[Any], Any]]] = None) -> (
              GradientTransformation[Any, Any]):
    return ChainedGradientTransformation([ScaleByAdam(b1, b2, eps, eps_root, mu_dtype),
                                          AddDecayedWeights(weight_decay, mask),
                                          Scale(-learning_rate)])


def sgd(learning_rate: float,
        momentum: Optional[float] = None,
        nesterov: bool = False,
        accumulator_dtype: Optional[Any] = None) -> GradientTransformation[Any, Any]:
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
    transforms = []
    if momentum is not None:
        transforms.append(Trace(decay=momentum, nesterov=nesterov,
                                accumulator_dtype=accumulator_dtype))
    transforms.append(Scale(-learning_rate))
    return ChainedGradientTransformation(transforms)
