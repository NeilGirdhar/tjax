import dataclasses
from dataclasses import MISSING, Field, asdict, astuple
from dataclasses import fields as d_fields
from dataclasses import is_dataclass, replace
from typing import (Any, Callable, Iterable, Mapping, MutableMapping, Optional, Tuple, TypeVar,
                    overload)

__all__ = ['field', 'Field', 'fields', 'asdict', 'astuple', 'replace', 'is_dataclass',
           'field_names', 'field_names_and_values', 'field_names_values_metadata', 'field_values',
           'document_dataclass']


T = TypeVar('T', bound=Any)


# NOTE: Actual return type is 'Field[T]', but we want to help type checkers
# to understand the magic that happens at runtime.
# pylint: disable=redefined-builtin
@overload  # `default` and `default_factory` are optional and mutually exclusive.
def field(*, static: bool = False, default: T, init: bool = ..., repr: bool = ...,
          hash: Optional[bool] = ..., compare: bool = ...,
          metadata: Optional[Mapping[str, Any]] = ...) -> T:
    ...


@overload
def field(*, static: bool = False, default_factory: Callable[[], T], init: bool = ...,
          repr: bool = ..., hash: Optional[bool] = ..., compare: bool = ...,
          metadata: Optional[Mapping[str, Any]] = ...) -> T:
    ...


@overload
def field(*, static: bool = False, init: bool = ..., repr: bool = ..., hash: Optional[bool] = ...,
          compare: bool = ..., metadata: Optional[Mapping[str, Any]] = ...) -> Any:
    ...


def field(*, static: bool = False, default: Any = MISSING,
          default_factory: Callable[[], Any] = MISSING, init: bool = True,  # type: ignore
          repr: bool = True, hash: Optional[bool] = None, compare: bool = True,
          metadata: Optional[Mapping[str, Any]] = None) -> Any:
    """
    Args:
        static: Indicates whether a field is a pytree or static.  Pytree fields are
            differentiated and traced.  Static fields are hashed and compared.
    """
    if metadata is None:
        metadata = {}
    return dataclasses.field(metadata={**metadata, 'static': static},
                             default=default, default_factory=default_factory, init=init, repr=repr,
                             hash=hash, compare=compare)  # type: ignore


def fields(d: Any, *, static: Optional[bool] = None) -> Iterable[Field[Any]]:
    if static is None:
        yield from d_fields(d)
    for this_field in d_fields(d):
        if this_field.metadata.get('static', False) == static:
            yield this_field


def field_names(d: Any, *, static: Optional[bool] = None) -> Iterable[str]:
    for this_field in fields(d, static=static):
        yield this_field.name


def field_names_and_values(d: Any, *, static: Optional[bool] = None) -> Iterable[Tuple[str, Any]]:
    for name in field_names(d, static=static):
        yield name, getattr(d, name)


def field_values(d: Any, *, static: Optional[bool] = None) -> Iterable[Any]:
    for name in field_names(d, static=static):
        yield getattr(d, name)


def field_names_values_metadata(d: Any, *, static: Optional[bool] = None) -> (
        Iterable[Tuple[str, Any, Mapping[str, Any]]]):
    for this_field in fields(d, static=static):
        yield this_field.name, getattr(d, this_field.name), this_field.metadata


def document_dataclass(pdoc: MutableMapping[str, Any], name: str) -> None:
    pdoc[f'{name}.static_fields'] = False
    pdoc[f'{name}.nonstatic_fields'] = False
    pdoc[f'{name}.tree_flatten'] = False
    pdoc[f'{name}.tree_unflatten'] = False
    pdoc[f'{name}.display'] = False
    pdoc[f'{name}.replace'] = False
