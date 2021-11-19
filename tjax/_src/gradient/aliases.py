from __future__ import annotations

from typing import Any, Callable, List, Optional, Union

from .chain import ChainedGradientTransformation
from .transform import GradientTransformation
from .transforms import AddDecayedWeights, Scale, ScaleByAdam, ScaleByRms, ScaleByStddev, Trace

__all__ = ['adam', 'adamw', 'rmsprop', 'sgd']


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


def rmsprop(learning_rate: float,
            decay: float = 0.9,
            eps: float = 1e-8,
            initial_scale: float = 0.,
            centered: bool = False,
            momentum: Optional[float] = None,
            nesterov: bool = False) -> GradientTransformation[Any, Any]:
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
    Returns:
        the corresponding `GradientTransformation`.
    """
    # pylint: enable=line-too-long
    trace: List[GradientTransformation[Any, Any]] = (
        [Trace(decay=momentum, nesterov=nesterov)] if momentum is not None else [])
    if centered:
        return ChainedGradientTransformation([
            ScaleByStddev(decay=decay, eps=eps, initial_scale=initial_scale),
            Scale(-learning_rate)] + trace)
    return ChainedGradientTransformation([
        ScaleByRms(decay=decay, eps=eps, initial_scale=initial_scale),
        Scale(-learning_rate)] + trace)


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
    transforms: List[GradientTransformation[Any, Any]] = []
    if momentum is not None:
        transforms.append(Trace(decay=momentum, nesterov=nesterov,
                                accumulator_dtype=accumulator_dtype))
    transforms.append(Scale(-learning_rate))
    return ChainedGradientTransformation(transforms)
