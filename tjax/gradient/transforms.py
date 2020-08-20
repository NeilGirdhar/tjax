from typing import Generic, Mapping, Optional, Tuple, Union

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
               parameters: Optional[Weights],
               meta_parameters: Optional[Mapping[Any, Numeric]] = None) -> (
                   Tuple[Weights, ScaleState]):
        x = self.replace_meta_parameters_with_defaults(meta_parameters)
        return scale(x.step_size).update(gradient, state, parameters, meta_parameters)


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
               parameters: Optional[Weights],
               meta_parameters: Optional[Mapping[Any, Numeric]] = None) -> (
                   Tuple[Weights, ScaleState]):
        x = self.replace_meta_parameters_with_defaults(meta_parameters)
        return scale_by_adam(x.beta1, x.beta2, x.epsilon, x.epsilon_root).update(
            gradient, state, parameters, meta_parameters)
