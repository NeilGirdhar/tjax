from typing import Any, Generic, List, Optional, Tuple, TypeVar

from chex import Numeric
from optax import ScaleByAdamState, ScaleState, scale, scale_by_adam

from .annotations import PyTree
from .dataclass import dataclass

Parameters = TypeVar('Parameters', bound=PyTree)
State = TypeVar('State', bound=PyTree)


__all__ = ['GradientTransformation', 'ChainedGradientTransformation', 'Scale', 'ScaleByAdam',
           'adam']


@dataclass
class GradientTransformation(Generic[State, Parameters]):

    def init(self, parameters: Parameters) -> State:
        raise NotImplementedError

    def update(self,
               gradient: Parameters,
               state: State,
               parameters: Optional[Parameters]) -> Tuple[Parameters, State]:
        raise NotImplementedError


@dataclass
class ChainedGradientTransformation(GradientTransformation[List[PyTree], Parameters],
                                    Generic[Parameters]):

    transforms: List[GradientTransformation[Any, Parameters]]

    def init(self, parameters: Parameters) -> List[PyTree]:
        return [transform.init(parameters)
                for transform in self.transforms]

    def update(self,
               gradient: Parameters,
               state: List[PyTree],
               parameters: Optional[Parameters]) -> Tuple[Parameters, List[PyTree]]:
        new_state: List[PyTree] = []
        for sub_state, transform in zip(state, self.transforms):
            gradient, new_state = transform.update(gradient, sub_state, parameters)
            new_state.append(new_state)
        return gradient, new_state


@dataclass
class Scale(GradientTransformation[ScaleState, Parameters], Generic[Parameters]):

    step_size: Numeric

    def init(self, parameters: Parameters) -> ScaleState:
        return scale(self.step_size).init(parameters)

    def update(self,
               gradient: Parameters,
               state: ScaleState,
               parameters: Optional[Parameters]) -> Tuple[Parameters, ScaleState]:
        return scale(self.step_size).update(gradient, state, parameters)


@dataclass
class ScaleByAdam(GradientTransformation[ScaleByAdamState, Parameters], Generic[Parameters]):

    beta1: Numeric = 0.9
    beta2: Numeric = 0.999
    epsilon: Numeric = 1e-8
    epsilon_root: Numeric = 0.0

    def init(self, parameters: Parameters) -> ScaleByAdamState:
        return scale_by_adam(self.beta1, self.beta2, self.epsilon, self.epsilon_root).init(
            parameters)

    def update(self,
               gradient: Parameters,
               state: ScaleByAdamState,
               parameters: Optional[Parameters]) -> Tuple[Parameters, ScaleByAdamState]:
        return scale_by_adam(self.beta1, self.beta2, self.epsilon, self.epsilon_root).update(
            gradient, state, parameters)


def adam(learning_rate: float,
         beta1: float = 0.9,
         beta2: float = 0.999,
         epsilon: float = 1e-8) -> GradientTransformation[Any, Any]:
    return ChainedGradientTransformation([ScaleByAdam(beta1, beta2, epsilon),
                                          Scale(-learning_rate)])
