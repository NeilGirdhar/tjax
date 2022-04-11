# pylint: disable=redefined-builtin
from __future__ import annotations

import dataclasses
import sys
from dataclasses import MISSING, fields
from typing import Any, Callable, Dict, Mapping, Optional, TypeVar, overload

__all__ = ['field', 'as_shallow_dict']


T = TypeVar('T', bound=Any)


if sys.version_info >= (3, 10):
    # NOTE: Actual return type is 'Field[T]', but we want to help type checkers
    # to understand the magic that happens at runtime.
    @overload  # `default` and `default_factory` are optional and mutually exclusive.
    def field(*, static: bool = False, default: T, init: bool = ..., repr: bool = ...,
              hash: Optional[bool] = ..., compare: bool = ...,
              metadata: Optional[Mapping[str, Any]] = ..., kw_only: bool = ...) -> T:
        ...

    @overload
    def field(*, static: bool = False, default_factory: Callable[[], T], init: bool = ...,
              repr: bool = ..., hash: Optional[bool] = ..., compare: bool = ...,
              metadata: Optional[Mapping[str, Any]] = ..., kw_only: bool = ...) -> T:
        ...

    @overload
    def field(*, static: bool = False, init: bool = ..., repr: bool = ...,
              hash: Optional[bool] = ..., compare: bool = ...,
              metadata: Optional[Mapping[str, Any]] = ..., kw_only: bool = ...) -> Any:
        ...

    def field(*, static: bool = False, default: Any = MISSING,
              default_factory: Any = MISSING, init: bool = True,
              repr: bool = True, hash: Optional[bool] = None, compare: bool = True,
              metadata: Optional[Mapping[str, Any]] = None,
              kw_only: Any = MISSING) -> Any:
        """
        Args:
            static: Indicates whether a field is a pytree or static.  Pytree fields are
                differentiated and traced.  Static fields are hashed and compared.
        """
        metadata_dict: Dict[str, Any] = {} if metadata is None else dict(metadata)
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
else:
    # NOTE: Actual return type is 'Field[T]', but we want to help type checkers
    # to understand the magic that happens at runtime.
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
    def field(*, static: bool = False, init: bool = ..., repr: bool = ...,
              hash: Optional[bool] = ..., compare: bool = ...,
              metadata: Optional[Mapping[str, Any]] = ...) -> Any:
        ...

    def field(*, static: bool = False, default: Any = MISSING,
              default_factory: Callable[[], Any] = MISSING, init: bool = True,
              repr: bool = True, hash: Optional[bool] = None, compare: bool = True,
              metadata: Optional[Mapping[str, Any]] = None) -> Any:
        """
        Args:
            static: Indicates whether a field is a pytree or static.  Pytree fields are
                differentiated and traced.  Static fields are hashed and compared.
        """
        metadata_dict: Dict[str, Any] = {} if metadata is None else dict(metadata)
        metadata_dict['static'] = static
        if default is MISSING:
            return dataclasses.field(default_factory=default_factory,
                                     init=init,
                                     repr=repr,
                                     hash=hash,
                                     compare=compare,
                                     metadata=metadata_dict)
        return dataclasses.field(default=default,
                                 init=init,
                                 repr=repr,
                                 hash=hash,
                                 compare=compare,
                                 metadata=metadata_dict)


def as_shallow_dict(dcls: Any) -> Dict[str, Any]:
    return {field.name: getattr(dcls, field.name) for field in fields(dcls)}
