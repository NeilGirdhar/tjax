from __future__ import annotations

import dataclasses
from dataclasses import MISSING
from typing import Any, Callable, Mapping, Optional, TypeVar, overload

__all__ = ['field']


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
