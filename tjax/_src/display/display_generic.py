from __future__ import annotations

import math
from collections.abc import Generator, Mapping, MutableSet
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from functools import singledispatch
from numbers import Number
from types import FunctionType
from typing import Any

import numpy as np
from jax import Array
from jax.errors import TracerArrayConversionError
from jax.tree_util import PyTreeDef
from jaxlib._jax import PjitFunction  # noqa: PLC2701
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from tjax.dataclasses import DataclassInstance

from ..annotations import NumpyArray
from .colors import solarized

assert isinstance(PyTreeDef, type)
assert isinstance(PjitFunction, type)

# Constants ----------------------------------------------------------------------------------------
# Numeric (warm)
_numpy_array_color = solarized['magenta']
_jax_array_color = solarized['orange']
_number_color = solarized['yellow']
# solarized['red']
# Classes (cool)
_class_color = solarized['violet']
# solarized['cyan']
_type_color = solarized['green']
_string_color = solarized['base0']
# Other
_key_color = solarized['blue']
_separator_color = solarized['base00']
_table_color = solarized['base02']
# _unknown_color = f"{solarized['red']} bold"
_seen_color = solarized['red']


# Extra imports ------------------------------------------------------------------------------------
def attribute_filter(value: object, attribute_name: str) -> bool:
    # is_private = attribute_name.startswith('_')
    return True


@singledispatch
def display_generic(value: object,
                    *,
                    seen: MutableSet[int] | None = None,
                    key: str = '',
                    ) -> Tree:
    if seen is None:
        seen = set()
    with _verify(value, seen, key) as x:
        if x:
            return x
        if is_dataclass(value) and not isinstance(value, type):
            return _display_dataclass(value, seen=seen, key=key)
        return _display_object(value, seen=seen, key=key)
    # _assemble(key, Text(str(value), style=_unknown_color))


@display_generic.register
def _(value: str,
      *,
      seen: MutableSet[int] | None = None,
      key: str = '',
      ) -> Tree:
    # No need for seen set ellision.
    return _assemble(key, Text(f'"{value}"', style=_string_color))


@display_generic.register(type)
def _(value: type[Any],
      *,
      seen: MutableSet[int] | None = None,
      key: str = '',
      ) -> Tree:
    # No need for seen set ellision.
    return _assemble(key, Text(f"type[{value.__name__}]", style=_type_color))


@display_generic.register(np.ndarray)
def _(value: NumpyArray,
      *,
      seen: MutableSet[int] | None = None,
      key: str = '',
      ) -> Tree:
    # No need for seen set ellision.
    retval = _assemble(key,
                       Text(f"NumPy Array {value.shape} {value.dtype}", style=_numpy_array_color))
    _show_array(retval, value)
    return retval


@display_generic.register
def _(value: Array,
      *,
      seen: MutableSet[int] | None = None,
      key: str = '',
      ) -> Tree:
    # No need for seen set ellision.
    retval = _assemble(key,
                       Text(f"Jax Array {value.shape} {value.dtype}",
                            style=_jax_array_color))
    try:
        np_value = np.asarray(value)
    except TracerArrayConversionError:
        pass  # This happens when trying to print a tracer in immediate mode.
    else:
        _show_array(retval, np_value)
    return retval


@display_generic.register(type(None))
@display_generic.register(Number)
def _(value: Number | None,
      *,
      seen: MutableSet[int] | None = None,
      key: str = '',
      ) -> Tree:
    # No need for seen set ellision.
    return _assemble(key, Text(str(value), style=_number_color))


@display_generic.register(Mapping)
def _(value: Mapping[Any, Any],
      *,
      seen: MutableSet[int] | None = None,
      key: str = '',
      ) -> Tree:
    if seen is None:
        seen = set()
    with _verify(value, seen, key) as x:
        if x:
            return x
        retval = display_class(key, type(value))
        for sub_key, sub_value in value.items():
            retval.children.append(display_generic(sub_value, seen=seen, key=str(sub_key)))
        return retval


@display_generic.register(tuple)
@display_generic.register(list)
def _(value: tuple[Any, ...] | list[Any],
      *,
      seen: MutableSet[int] | None = None,
      key: str = '',
      ) -> Tree:
    if seen is None:
        seen = set()
    with _verify(value, seen, key) as x:
        if x:
            return x
        retval = display_class(key, type(value))
        for sub_value in value:
            retval.children.append(display_generic(sub_value, seen=seen, key=""))
        return retval


@display_generic.register(PyTreeDef)
def _(value: object,
      *,
      seen: MutableSet[int] | None = None,
      key: str = '',
      ) -> Tree:
    if seen is None:
        seen = set()
    with _verify(value, seen, key) as x:
        if x:
            return x
        retval = display_class(key, type(value))
        retval.children.append(display_generic(hash(value), seen=seen, key="hash"))
        return retval


@display_generic.register(FunctionType)
@display_generic.register(PjitFunction)
def _(value: object,
      *,
      seen: MutableSet[int] | None = None,
      key: str = '',
      ) -> Tree:
    if seen is None:
        seen = set()
    with _verify(value, seen, key) as x:
        if x:
            return x
        name = getattr(value, '__qualname__', "")
        retval = display_class(key, type(value))
        retval.children.append(display_generic(name, seen=seen, key="name"))
        return retval


# Public unexported functions ----------------------------------------------------------------------
def display_class(key: str, cls: type[Any]) -> Tree:
    name = cls.__name__
    tags = []
    if is_dataclass(cls):
        tags.append('dataclass')
    if tags:
        name += f"[{','.join(tags)}]"
    return _assemble(key, Text(name, style=_class_color))


# Private functions --------------------------------------------------------------------------------
def _display_dataclass(value: DataclassInstance,
                       *,
                       seen: MutableSet[int],
                       key: str = '',
                       ) -> Tree:
    retval = display_class(key, type(value))
    names = set()
    for field_info in fields(value):
        name = field_info.name
        names.add(name)
        if not field_info.repr:
            continue
        if not attribute_filter(value, name):
            continue
        display_name = name
        if not field_info.init:
            display_name += ' (module)'
        sub_value = getattr(value, name, None)
        retval.children.append(display_generic(sub_value, seen=seen, key=display_name))
    variables = _variables(value)
    for name, sub_value in variables.items():
        if name in names or not attribute_filter(value, name):
            continue
        retval.children.append(display_generic(sub_value, seen=seen, key=name + '*'))
    return retval


def _display_object(value: object,
                    *,
                    seen: MutableSet[int],
                    key: str = '',
                    ) -> Tree:
    retval = display_class(key, type(value))
    variables = _variables(value)
    for name, sub_value in variables.items():
        if not attribute_filter(value, name):
            continue
        retval.children.append(display_generic(sub_value, seen=seen, key=name))
    return retval


def _variables(value: object) -> dict[str, Any]:
    try:
        variables = vars(value)
    except TypeError:
        variables = ({name: getattr(value, name)
                      for name in value.__slots__  # pyright: ignore
                      if hasattr(value, name)}  # Work around a Jax oddity.
                     if hasattr(value, '__slots__')
                     else {})
    return {key: value
            for key, value in variables.items()
            if key != '_module__state'
            if not key.startswith('__')}


@contextmanager
def _verify(value: object,
            seen: MutableSet[int],
            key: str
            ) -> Generator[Tree | None]:
    if id(value) in seen:
        yield _assemble(key, Text(f'<seen {id(value)}>', style=_seen_color))
        return
    seen.add(id(value))
    yield None
    seen.remove(id(value))


def _assemble(key: str,
              type_text: Text,
              separator: str = '=') -> Tree:
    """Returns: A Rich Tree for a given key-value pair."""
    if key:
        return Tree(Text.assemble(Text(key, style=_key_color),
                                  Text(separator, style=_separator_color),
                                  type_text))
    return Tree(type_text)


@singledispatch
def _format_number(x: object) -> str:
    return str(x)


@_format_number.register(np.inexact)
def _(x: np.inexact[Any]) -> str:
    return f"{x:.4f}"


def _show_array(tree: Tree, array: NumpyArray) -> None:
    """Add a representation of array to the Rich tree."""
    if not issubclass(array.dtype.type, np.bool_ | np.number):
        return
    if math.prod(array.shape) == 0:
        return
    if 1 in array.shape:
        _show_array(tree,
                    array[tuple[int | slice, ...](0 if s == 1 else slice(None)
                                                        for s in array.shape)])
        return
    if any(x > 12 for x in array.shape) or len(array.shape) > 2:  # noqa: PLR2004
        xarray = np.asarray(array)
        tree.children.append(display_generic(np.mean(xarray).item(), seen=set(), key="mean"))
        tree.children.append(display_generic(np.std(xarray).item(), seen=set(), key="deviation"))
        return
    if len(array.shape) == 0:
        _ = tree.add(_format_number(array[()]))
        return
    table = Table(show_header=False,
                  show_edge=False,
                  style=_table_color)
    for _ in range(array.shape[-1]):
        table.add_column()
    if len(array.shape) == 1:
        table.add_row(*(_format_number(array[i])
                        for i in range(array.shape[0])))
    elif len(array.shape) == 2:  # noqa: PLR2004
        for j in range(array.shape[0]):
            table.add_row(*(_format_number(array[j, i])
                            for i in range(array.shape[1])))
    _ = tree.add(table)
