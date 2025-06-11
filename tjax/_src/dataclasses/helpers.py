from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import MISSING, fields
from typing import Any, ClassVar, Never, Protocol, TypeVar, overload, runtime_checkable

T = TypeVar('T')


@runtime_checkable
class DataclassInstance(Protocol):
    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[Any]]]


# NOTE: Actual return type is 'Field[T]', but we want to help type checkers
# to understand the magic that happens at runtime.
@overload  # `default` and `default_factory` are optional and mutually exclusive.
def field(*, static: bool = False, default: T, init: bool = ...,
          repr: bool = ...,
          hash: bool | None = ...,
          compare: bool = ...,
          metadata: Mapping[str, Any] | None = ..., kw_only: bool = ...) -> T:
    ...


@overload
def field(*, static: bool = False, default_factory: Callable[[], T], init: bool = ...,
          repr: bool = ...,
          hash: bool | None = ...,
          compare: bool = ...,
          metadata: Mapping[str, Any] | None = ..., kw_only: bool = ...) -> T:
    ...


@overload
def field(*, static: bool = False, init: bool = ...,
          repr: bool = ...,
          hash: bool | None = ...,
          compare: bool = ...,
          metadata: Mapping[str, Any] | None = ..., kw_only: bool = ...) -> Never:
    ...


def field(*, static: bool = False, default: Any = MISSING,
          default_factory: Any = MISSING, init: bool = True,
          repr: bool = True,  # noqa: A002
          hash: bool | None = None,  # noqa: A002
          compare: bool = True,
          metadata: Mapping[str, Any] | None = None,
          kw_only: Any = MISSING) -> Any:
    """A field creator with a static indicator.

    The static flag indicates whether a field is a pytree or static.  Pytree fields are
    differentiated and traced.  Static fields are hashed and compared.
    """
    metadata_dict: dict[str, Any] = {} if metadata is None else dict(metadata)
    metadata_dict['static'] = static
    if default is MISSING:
        return dataclasses.field(default_factory=default_factory,
                                 init=init,
                                 repr=repr,
                                 hash=hash,
                                 compare=compare,
                                 metadata=metadata_dict,
                                 kw_only=kw_only)
    return dataclasses.field(default=default,
                             init=init,
                             repr=repr,
                             hash=hash,
                             compare=compare,
                             metadata=metadata_dict,
                             kw_only=kw_only)


def as_shallow_dict(dcls: DataclassInstance) -> dict[str, Any]:
    return {field.name: getattr(dcls, field.name) for field in fields(dcls)}
