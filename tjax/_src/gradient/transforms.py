from typing import Generic, Optional, Tuple

from optax import ScaleByAdamState, ScaleState, additive_weight_decay, scale, scale_by_adam

from ..annotations import RealNumeric
from ..dataclasses import dataclass
from .transform import GradientTransformation, Weights

__all__ = ['Scale', 'ScaleByAdam', 'AdditiveWeightDecay']


@dataclass
class Scale(GradientTransformation[ScaleState, Weights], Generic[Weights]):
    step_size: RealNumeric

    def init(self, parameters: Weights) -> ScaleState:
        return scale(self.step_size).init(parameters)

    def update(self,
               gradient: Weights,
               state: ScaleState,
               parameters: Optional[Weights]) -> Tuple[Weights, ScaleState]:
        return scale(self.step_size).update(gradient, state, parameters)


@dataclass
class ScaleByAdam(GradientTransformation[ScaleByAdamState, Weights], Generic[Weights]):
    beta1: RealNumeric = 0.9
    beta2: RealNumeric = 0.999
    epsilon: RealNumeric = 1e-8
    epsilon_root: RealNumeric = 0.0

    def init(self, parameters: Weights) -> ScaleByAdamState:
        return scale_by_adam(self.beta1, self.beta2, self.epsilon, self.epsilon_root).init(
            parameters)

    def update(self,
               gradient: Weights,
               state: ScaleByAdamState,
               parameters: Optional[Weights]) -> Tuple[Weights, ScaleByAdamState]:
        return scale_by_adam(self.beta1, self.beta2, self.epsilon, self.epsilon_root).update(
            gradient, state, parameters)


@dataclass
class AdditiveWeightDecay(GradientTransformation[None, Weights], Generic[Weights]):
    weight_decay: RealNumeric = 0.0

    def init(self, parameters: Weights) -> None:
        return None

    def update(self,
               gradient: Weights,
               state: None,
               parameters: Optional[Weights]) -> Tuple[Weights, ScaleState]:
        return additive_weight_decay(self.weight_decay).update(gradient, state, parameters)
