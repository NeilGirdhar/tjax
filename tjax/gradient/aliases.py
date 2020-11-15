from typing import Any

from .chain import ChainedGradientTransformation
from .transform import GradientTransformation
from .transforms import AdditiveWeightDecay, Scale, ScaleByAdam

__all__ = ['adam', 'adamw']


def adam(learning_rate: float,
         beta1: float = 0.9,
         beta2: float = 0.999,
         epsilon: float = 1e-8) -> GradientTransformation[Any, Any]:
    return ChainedGradientTransformation([ScaleByAdam(beta1=beta1, beta2=beta2, epsilon=epsilon),
                                          Scale(-learning_rate)])


def adamw(learning_rate: float,
          beta1: float = 0.9,
          beta2: float = 0.999,
          epsilon: float = 1e-8,
          weight_decay: float = 1e-4) -> GradientTransformation[Any, Any]:
    return ChainedGradientTransformation([ScaleByAdam(beta1=beta1, beta2=beta2, epsilon=epsilon),
                                          AdditiveWeightDecay(weight_decay),
                                          Scale(-learning_rate)])
