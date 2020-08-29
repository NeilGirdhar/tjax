from typing import Generic, Optional, Tuple, Union

from chex import Numeric
from optax import ScaleByAdamState, ScaleState, scale, scale_by_adam

from ..dataclass import dataclass
from .meta_parameter import MetaParameter
from .transform import GradientTransformation, Weights

__all__ = ['Scale', 'ScaleByAdam']


NumericOrMeta = Union[Numeric, MetaParameter]


@dataclass
class Scale(GradientTransformation[ScaleState, Weights], Generic[Weights]):

    step_size: NumericOrMeta

    def init(self, parameters: Weights) -> ScaleState:
        return scale(self.step_size).init(parameters)

    def update(self,
               gradient: Weights,
               state: ScaleState,
               parameters: Optional[Weights]) -> Tuple[Weights, ScaleState]:
        return scale(self.step_size).update(gradient, state, parameters)


@dataclass
class ScaleByAdam(GradientTransformation[ScaleByAdamState, Weights], Generic[Weights]):

    beta1: NumericOrMeta = 0.9
    beta2: NumericOrMeta = 0.999
    epsilon: NumericOrMeta = 1e-8
    epsilon_root: NumericOrMeta = 0.0

    def init(self, parameters: Weights) -> ScaleByAdamState:
        return scale_by_adam(self.beta1, self.beta2, self.epsilon, self.epsilon_root).init(
            parameters)

    def update(self,
               gradient: Weights,
               state: ScaleByAdamState,
               parameters: Optional[Weights]) -> Tuple[Weights, ScaleState]:
        return scale_by_adam(self.beta1, self.beta2, self.epsilon, self.epsilon_root).update(
            gradient, state, parameters)
