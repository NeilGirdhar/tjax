from typing import Any

from .chain import ChainedGradientTransformation
from .transform import GradientTransformation
from .transforms import Scale, ScaleByAdam

__all__ = ['adam']


def adam(learning_rate: float,
         beta1: float = 0.9,
         beta2: float = 0.999,
         epsilon: float = 1e-8) -> GradientTransformation[Any, Any]:
    return ChainedGradientTransformation([ScaleByAdam(beta1, beta2, epsilon),
                                          Scale(-learning_rate)])
