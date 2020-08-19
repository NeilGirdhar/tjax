from typing import Callable, Dict, Generic, Mapping, Optional, Tuple, TypeVar, Union

from chex import Numeric

from ..annotations import PyTree
from ..dataclass import dataclass
from .meta_parameter import MetaParameter

__all__ = ['GradientTransformation']


Parameters = TypeVar('Parameters', bound=PyTree)
State = TypeVar('State', bound=PyTree)


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

    def replace_meta_parameters(self: T,
                                f: Callable[[MetaParameter], Union[Numeric, MetaParameter]]) -> T:
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
