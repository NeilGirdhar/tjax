from typing import Generic, Optional, Tuple

from chex import Numeric
from optax import ScaleByAdamState, ScaleState, scale, scale_by_adam

from ..dataclass import dataclass
from .transform import GradientTransformation, Weights

__all__ = ['Scale', 'ScaleByAdam']


@dataclass
class Scale(GradientTransformation[ScaleState, Weights], Generic[Weights]):

    step_size: Numeric

    def init(self, parameters: Weights) -> ScaleState:
        return scale(self.step_size).init(parameters)

    def update(self,
               gradient: Weights,
               state: ScaleState,
               parameters: Optional[Weights]) -> Tuple[Weights, ScaleState]:
        return scale(self.step_size).update(gradient, state, parameters)


@dataclass
class ScaleByAdam(GradientTransformation[ScaleByAdamState, Weights], Generic[Weights]):

    beta1: Numeric = 0.9
    beta2: Numeric = 0.999
    epsilon: Numeric = 1e-8
    epsilon_root: Numeric = 0.0

    def init(self, parameters: Weights) -> ScaleByAdamState:
        return scale_by_adam(self.beta1, self.beta2, self.epsilon, self.epsilon_root).init(
            parameters)

    def update(self,
               gradient: Weights,
               state: ScaleByAdamState,
               parameters: Optional[Weights]) -> Tuple[Weights, ScaleState]:
        return scale_by_adam(self.beta1, self.beta2, self.epsilon, self.epsilon_root).update(
            gradient, state, parameters)
