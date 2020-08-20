from typing import Generic, Hashable, Mapping, Optional, Tuple, Union

from chex import Numeric
from optax import ScaleByAdamState, ScaleState, scale, scale_by_adam

from ..dataclass import dataclass
from .meta_parameter import MetaParameter
from .transform import GradientTransformation, Parameters

__all__ = ['Scale', 'ScaleByAdam']


NumericOrMeta = Union[Numeric, MetaParameter]


@dataclass
class Scale(GradientTransformation[ScaleState, Parameters], Generic[Parameters]):

    step_size: NumericOrMeta

    def init(self, parameters: Parameters) -> ScaleState:
        return scale(self.step_size).init(parameters)

    def update(self,
               gradient: Parameters,
               state: ScaleState,
               parameters: Optional[Parameters],
               meta_parameters: Optional[Mapping[Hashable, Numeric]] = None) -> (
                   Tuple[Parameters, ScaleState]):
        x = self.replace_meta_parameters_with_defaults(meta_parameters)
        return scale(x.step_size).update(gradient, state, parameters)


@dataclass
class ScaleByAdam(GradientTransformation[ScaleByAdamState, Parameters], Generic[Parameters]):

    beta1: NumericOrMeta = 0.9
    beta2: NumericOrMeta = 0.999
    epsilon: NumericOrMeta = 1e-8
    epsilon_root: NumericOrMeta = 0.0

    def init(self, parameters: Parameters) -> ScaleByAdamState:
        return scale_by_adam(self.beta1, self.beta2, self.epsilon, self.epsilon_root).init(
            parameters)

    def update(self,
               gradient: Parameters,
               state: ScaleByAdamState,
               parameters: Optional[Parameters],
               meta_parameters: Optional[Mapping[Hashable, Numeric]] = None) -> (
                   Tuple[Parameters, ScaleState]):
        x = self.replace_meta_parameters_with_defaults(meta_parameters)
        return scale_by_adam(x.beta1, x.beta2, x.epsilon, x.epsilon_root).update(
            gradient, state, parameters)
