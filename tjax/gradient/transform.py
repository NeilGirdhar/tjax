from typing import Any, Callable, Dict, Generic, Mapping, Optional, Tuple, TypeVar, Union

from chex import Numeric

from ..annotations import PyTree
from ..dataclass import dataclass, fields
from .meta_parameter import MetaParameter

__all__ = ['GradientTransformation', 'GradientState']


GradientState = PyTree
Weights = TypeVar('Weights', bound=PyTree)
State = TypeVar('State', bound=PyTree)


@dataclass
class GradientTransformation(Generic[State, Weights]):

    T = TypeVar('T', bound='GradientTransformation[State, Weights]')

    def init(self, parameters: Weights) -> State:
        raise NotImplementedError

    def update(self,
               gradient: Weights,
               state: State,
               parameters: Optional[Weights]) -> Tuple[Weights, State]:
        raise NotImplementedError

    def has_meta_parameters(self) -> bool:
        return any(isinstance(getattr(self, field.name), MetaParameter)
                   for field in fields(self))

    def replace_meta_parameters(self: T,
                                f: Callable[[MetaParameter], Union[Numeric, MetaParameter]]) -> T:
        replacements: Dict[str, Numeric] = {}
        for field in fields(self):
            name = field.name
            value = getattr(self, name)
            if isinstance(value, MetaParameter):
                replacements[name] = f(value)
        return self.replace(**replacements)

    def replace_meta_parameters_with_defaults(
            self: T,
            meta_parameters: Optional[Mapping[Any, Numeric]] = None) -> T:
        if meta_parameters is None:
            return self

        def f(meta_parameter: MetaParameter) -> Numeric:
            # https://github.com/python/mypy/issues/2608
            return meta_parameters[meta_parameter.key]  # type: ignore
        return self.replace_meta_parameters(f)
