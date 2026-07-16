from __future__ import annotations

import dataclasses
import sys
from collections.abc import Callable, Mapping
from dataclasses import _MISSING_TYPE, fields
from typing import Any, ClassVar, Protocol, overload, runtime_checkable


@runtime_checkable
class DataclassInstance(Protocol):
    """Protocol satisfied by any object created by :func:`dataclasses.dataclass`."""

    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[Any]]]


# NOTE: Actual return type is 'Field[T]', but we want to help type checkers
# to understand the magic that happens at runtime.
if sys.version_info >= (3, 14):

    @overload  # `default` and `default_factory` are optional and mutually exclusive.
    def field[T](
        *,
        static: bool = False,
        default: T,
        default_factory: _MISSING_TYPE = ...,
        init: bool = True,
        repr: bool = True,
        hash: bool | None = None,
        compare: bool = True,
        metadata: Mapping[Any, Any] | None = None,
        kw_only: bool | _MISSING_TYPE = ...,
        doc: str | None = None,
    ) -> T: ...
    @overload
    def field[T](
        *,
        static: bool = False,
        default: _MISSING_TYPE = ...,
        default_factory: Callable[[], T],
        init: bool = True,
        repr: bool = True,
        hash: bool | None = None,
        compare: bool = True,
        metadata: Mapping[Any, Any] | None = None,
        kw_only: bool | _MISSING_TYPE = ...,
        doc: str | None = None,
    ) -> T: ...
    @overload
    def field(
        *,
        static: bool = False,
        default: _MISSING_TYPE = ...,
        default_factory: _MISSING_TYPE = ...,
        init: bool = True,
        repr: bool = True,
        hash: bool | None = None,
        compare: bool = True,
        metadata: Mapping[Any, Any] | None = None,
        kw_only: bool | _MISSING_TYPE = ...,
        doc: str | None = None,
    ) -> Any: ...  # ruff:ignore[any-type]
    def field(
        *,
        static: bool = False,
        default: Any = dataclasses.MISSING,
        default_factory: Any = dataclasses.MISSING,
        init: bool = True,
        repr: bool = True,  # ruff:ignore[builtin-argument-shadowing]
        hash: bool | None = None,  # ruff:ignore[builtin-argument-shadowing]
        compare: bool = True,
        metadata: Mapping[str, Any] | None = None,
        kw_only: Any = dataclasses.MISSING,
        doc: str | None = None,
    ) -> Any:
        """A field creator with a static indicator.

        The static flag indicates whether a field is a pytree or static.  Pytree fields are
        differentiated and traced.  Static fields are hashed and compared.
        """
        metadata_dict: dict[str, Any] = {} if metadata is None else dict(metadata)
        metadata_dict["static"] = static
        if default is dataclasses.MISSING:
            return dataclasses.field(
                default_factory=default_factory,
                init=init,
                repr=repr,
                hash=hash,
                compare=compare,
                metadata=metadata_dict,
                kw_only=kw_only,
                doc=doc,
            )
        return dataclasses.field(
            default=default,
            init=init,
            repr=repr,
            hash=hash,
            compare=compare,
            metadata=metadata_dict,
            kw_only=kw_only,
            doc=doc,
        )
else:

    @overload  # `default` and `default_factory` are optional and mutually exclusive.
    def field[T](
        *,
        static: bool = False,
        default: T,
        default_factory: _MISSING_TYPE = ...,
        init: bool = True,
        repr: bool = True,
        hash: bool | None = None,
        compare: bool = True,
        metadata: Mapping[Any, Any] | None = None,
        kw_only: bool | _MISSING_TYPE = ...,
    ) -> T: ...
    @overload
    def field[T](
        *,
        static: bool = False,
        default: _MISSING_TYPE = ...,
        default_factory: Callable[[], T],
        init: bool = True,
        repr: bool = True,
        hash: bool | None = None,
        compare: bool = True,
        metadata: Mapping[Any, Any] | None = None,
        kw_only: bool | _MISSING_TYPE = ...,
    ) -> T: ...
    @overload
    def field(
        *,
        static: bool = False,
        default: _MISSING_TYPE = ...,
        default_factory: _MISSING_TYPE = ...,
        init: bool = True,
        repr: bool = True,
        hash: bool | None = None,
        compare: bool = True,
        metadata: Mapping[Any, Any] | None = None,
        kw_only: bool | _MISSING_TYPE = ...,
    ) -> Any: ...  # ruff:ignore[any-type]
    def field(
        *,
        static: bool = False,
        default: Any = dataclasses.MISSING,
        default_factory: Any = dataclasses.MISSING,
        init: bool = True,
        repr: bool = True,  # ruff:ignore[builtin-argument-shadowing]
        hash: bool | None = None,  # ruff:ignore[builtin-argument-shadowing]
        compare: bool = True,
        metadata: Mapping[str, Any] | None = None,
        kw_only: Any = dataclasses.MISSING,
    ) -> Any:
        """A field creator with a static indicator.

        The static flag indicates whether a field is a pytree or static.  Pytree fields are
        differentiated and traced.  Static fields are hashed and compared.
        """
        metadata_dict: dict[str, Any] = {} if metadata is None else dict(metadata)
        metadata_dict["static"] = static
        if default is dataclasses.MISSING:
            return dataclasses.field(
                default_factory=default_factory,
                init=init,
                repr=repr,
                hash=hash,
                compare=compare,
                metadata=metadata_dict,
                kw_only=kw_only,
            )
        return dataclasses.field(
            default=default,
            init=init,
            repr=repr,
            hash=hash,
            compare=compare,
            metadata=metadata_dict,
            kw_only=kw_only,
        )


def as_shallow_dict(dcls: DataclassInstance) -> dict[str, Any]:
    """Return a shallow ``{field_name: value}`` dict for a dataclass instance.

    Unlike :func:`dataclasses.asdict`, this does **not** recurse into nested
    dataclasses or containers, so the values are the live objects rather than
    deep copies.
    """
    return {field.name: getattr(dcls, field.name) for field in fields(dcls)}
