from typing import Any, Callable, Dict, Generic, List, Mapping, Optional, Tuple, TypeVar, Union

from chex import Numeric
from optax import ScaleByAdamState, ScaleState, scale, scale_by_adam

from .annotations import PyTree
from .dataclass import dataclass, field

Parameters = TypeVar('Parameters', bound=PyTree)
State = TypeVar('State', bound=PyTree)


__all__ = ['MetaParameter', 'GradientTransformation', 'ChainedGradientTransformation', 'Scale',
           'ScaleByAdam', 'adam']


@dataclass
class MetaParameter:

    name: str = field(static=True)


NumericOrMeta = Union[Numeric, MetaParameter]


def replace(value: NumericOrMeta,
            meta_parameters: Optional[Mapping[str, Numeric]]) -> Numeric:
    if isinstance(value, MetaParameter):
        if meta_parameters is None:
            raise ValueError
        return meta_parameters[value.name]
    return value


@dataclass
class GradientTransformation(Generic[State, Parameters]):

    T = TypeVar('T', bound='GradientTransformation[State, Parameters]')

    def init(self, parameters: Parameters) -> State:
        raise NotImplementedError

    def update(self,
               gradient: Parameters,
               state: State,
               parameters: Optional[Parameters]) -> Tuple[Parameters, State]:
        raise NotImplementedError

    def replace_meta_parameters(self: T, f: Callable[[MetaParameter], NumericOrMeta]) -> T:
        replacements: Dict[str, Numeric] = {}
        for name in self.__dataclass_fields__:  # type: ignore
            value = getattr(self, name)
            if isinstance(value, MetaParameter):
                replacements[name] = f(value)
        return self.replace(**replacements)

    def replace_meta_parameters_with_defaults(
            self: T,
            meta_parameters: Optional[Mapping[str, Numeric]] = None) -> T:
        if meta_parameters is None:
            return self

        def f(meta_parameter: MetaParameter) -> Numeric:
            # https://github.com/python/mypy/issues/2608
            return meta_parameters[meta_parameter.name]  # type: ignore
        return self.replace_meta_parameters(f)

    def rename_meta_parameters(self: T, prefix: str) -> T:
        if not prefix:
            return self

        def f(meta_parameter: MetaParameter) -> MetaParameter:
            return MetaParameter(prefix + meta_parameter.name)
        return self.replace_meta_parameters(f)


@dataclass
class ChainedGradientTransformation(GradientTransformation[List[PyTree], Parameters],
                                    Generic[Parameters]):

    U = TypeVar('U', bound='ChainedGradientTransformation[Parameters]')

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

    def replace_meta_parameters(self: U,  # type: ignore
                                f: Callable[[MetaParameter], NumericOrMeta]) -> U:
        new_transforms = []
        for transform in self.transforms:
            new_transforms.append(transform.replace_meta_parameters(f))
        return self.replace(transforms=new_transforms)


@dataclass
class Scale(GradientTransformation[ScaleState, Parameters], Generic[Parameters]):

    step_size: NumericOrMeta

    def init(self, parameters: Parameters) -> ScaleState:
        return scale(self.step_size).init(parameters)

    def update(self,
               gradient: Parameters,
               state: ScaleState,
               parameters: Optional[Parameters],
               meta_parameters: Optional[Mapping[str, Numeric]] = None) -> (
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
               meta_parameters: Optional[Mapping[str, Numeric]] = None) -> (
                   Tuple[Parameters, ScaleState]):
        x = self.replace_meta_parameters_with_defaults(meta_parameters)
        return scale_by_adam(x.beta1, x.beta2, x.epsilon, x.epsilon_root).update(
            gradient, state, parameters)


def adam(learning_rate: float,
         beta1: float = 0.9,
         beta2: float = 0.999,
         epsilon: float = 1e-8) -> GradientTransformation[Any, Any]:
    return ChainedGradientTransformation([ScaleByAdam(beta1, beta2, epsilon),
                                          Scale(-learning_rate)])
