from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableSet
from dataclasses import fields, is_dataclass
from functools import singledispatch
from numbers import Number
from typing import Any, List, Optional, Tuple, Type, Union

import numpy as np
from jax import Array
from jax.errors import TracerArrayConversionError
from jax.interpreters.batching import BatchTracer
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from ..annotations import NumpyArray
from .batch_dimensions import BatchDimensionIterator, BatchDimensions
from .colors import solarized

__all__ = ['display_generic', 'display_class']


# Constants ----------------------------------------------------------------------------------------
# Numeric (warm)
_numpy_array_color = solarized['magenta']
_jax_array_color = solarized['red']
_batch_array_color = solarized['orange']
_number_color = solarized['yellow']
# Classes (cool)
_class_color = solarized['violet']
_module_color = solarized['cyan']
_type_color = solarized['green']
_string_color = solarized['base0']
# Other
_key_color = solarized['blue']
_separator_color = solarized['base00']
_table_color = solarized['base02']
_unknown_color = f"{solarized['red']} bold"
_seen_color = solarized['red']


# Extra imports ------------------------------------------------------------------------------------
FlaxModule: Type[Any]
try:
    from flax.linen import Module as FlaxModule
    flax_loaded = True
except ImportError:
    flax_loaded = False
    FlaxModule = type(None)


@singledispatch
def display_generic(value: Any,
                    seen: MutableSet[int],
                    show_values: bool = True,
                    key: str = '',
                    batch_dims: BatchDimensions = None) -> Tree:
    if (x := _verify(value, seen, key)) is not None:
        return x
    if is_dataclass(value) and not isinstance(value, type):
        return display_dataclass(value, seen, show_values, key, batch_dims)
    return _assemble(key, Text(str(value), style=_unknown_color))


@display_generic.register
def _(value: str,
      seen: MutableSet[int],
      show_values: bool = True,
      key: str = '',
      batch_dims: Optional[Tuple[Optional[int], ...]] = None) -> Tree:
    if (x := _verify(value, seen, key)) is not None:
        return x
    return _assemble(key, Text(f"\"{value}\"", style=_string_color))


@display_generic.register(type)
def _(value: Type[Any],
      seen: MutableSet[int],
      show_values: bool = True,
      key: str = '',
      batch_dims: BatchDimensions = None) -> Tree:
    if (x := _verify(value, seen, key)) is not None:
        return x
    return _assemble(key, Text(f"type[{value.__name__}]", style=_type_color))


@display_generic.register
def _(value: BatchTracer,
      seen: MutableSet[int],
      show_values: bool = True,
      key: str = '',
      batch_dims: BatchDimensions = None) -> Tree:
    if (x := _verify(value, seen, key)) is not None:
        return x
    return _assemble(key,
                     Text(f"BatchTracer {value.shape} {value.dtype} "
                          f"batched over {value.val.shape[value.batch_dim]}",
                          style=_batch_array_color))


@display_generic.register(np.ndarray)
def _(value: NumpyArray,
      seen: MutableSet[int],
      show_values: bool = True,
      key: str = '',
      batch_dims: BatchDimensions = None) -> Tree:
    if (x := _verify(value, seen, key)) is not None:
        return x
    retval = _assemble(key,
                       Text(f"NumPy Array {value.shape} {value.dtype}{_batch_str(batch_dims)}",
                            style=_numpy_array_color))
    _show_array(retval, value)
    return retval


@display_generic.register
def _(value: Array,
      seen: MutableSet[int],
      show_values: bool = True,
      key: str = '',
      batch_dims: BatchDimensions = None) -> Tree:
    if (x := _verify(value, seen, key)) is not None:
        return x
    retval = _assemble(key,
                       Text(f"Jax Array {value.shape} {value.dtype}", style=_jax_array_color))
    try:
        np_value = np.asarray(value)
    except TracerArrayConversionError:
        pass
    else:
        _show_array(retval, np_value)
    return retval


@display_generic.register(type(None))
@display_generic.register(Number)
def _(value: Union[None, Number],
      seen: MutableSet[int],
      show_values: bool = True,
      key: str = '',
      batch_dims: BatchDimensions = None) -> Tree:
    if (x := _verify(value, seen, key)) is not None:
        return x
    return _assemble(key, Text(str(value), style=_number_color))


@display_generic.register(Mapping)
def _(value: Mapping[Any, Any],
      seen: MutableSet[int],
      show_values: bool = True,
      key: str = '',
      batch_dims: BatchDimensions = None) -> Tree:
    if (x := _verify(value, seen, key)) is not None:
        return x
    retval = display_class(key, type(value))
    for sub_batch_dims, (sub_key, sub_value) in zip(_batch_dimension_iterator(value.values(),
                                                                              batch_dims),
                                                    value.items()):
        retval.children.append(display_generic(sub_value, seen, show_values, sub_key,
                                               sub_batch_dims))
    return retval


@display_generic.register(tuple)
@display_generic.register(list)
def _(value: Union[Tuple[Any, ...], List[Any]],
      seen: MutableSet[int],
      show_values: bool = True,
      key: str = '',
      batch_dims: BatchDimensions = None) -> Tree:
    if (x := _verify(value, seen, key)) is not None:
        return x
    retval = display_class(key, type(value))
    for sub_batch_dims, sub_value in zip(_batch_dimension_iterator(value, batch_dims), value):
        retval.children.append(display_generic(sub_value, seen, show_values, "", sub_batch_dims))
    return retval


def display_dataclass(value: Any,
                      seen: MutableSet[int],
                      show_values: bool = True,
                      key: str = '',
                      batch_dims: BatchDimensions = None) -> Tree:
    is_module = flax_loaded and isinstance(value, FlaxModule)
    retval = display_class(key, type(value), is_module)
    bdi = BatchDimensionIterator(batch_dims)
    for field_info in fields(value):
        name = field_info.name
        if is_module and name in {'parent', 'name'}:
            continue
        sub_value = getattr(value, name)
        sub_batch_dims = bdi.advance(sub_value)
        retval.children.append(display_generic(sub_value, seen, show_values, name, sub_batch_dims))
    if is_module:
        retval.children.append(display_generic(value.name, seen, show_values, 'name', None))
        retval.children.append(display_generic(value.parent is not None, seen, show_values,
                                               'has_parent', None))
        retval.children.append(display_generic(value.scope is not None, seen, show_values, 'bound',
                                               None))
        # pylint: disable=protected-access
        for name, child_module in value._state.children.items():  # pytype: disable=attribute-error
            if not isinstance(child_module, FlaxModule):
                continue
            retval.children.append(display_generic(child_module, seen, show_values, name, None))
    return retval


# Public unexported functions ----------------------------------------------------------------------
def display_class(key: str, cls: Type[Any], is_module: bool = False) -> Tree:
    type_color = _module_color if is_module else _class_color
    return _assemble(key, Text(cls.__name__, style=type_color))


def _verify(value: Any,
            seen: MutableSet[int],
            key: str) -> Optional[Tree]:
    if id(value) in seen:
        return _assemble(key, Text('<seen>', style=_seen_color))
    if is_dataclass(value) and not isinstance(value, type):
        seen.add(id(value))
    return None


# Private functions --------------------------------------------------------------------------------
def _assemble(key: str,
              type_text: Text,
              separator: str = '=') -> Tree:
    if key:
        return Tree(Text.assemble(Text(key, style=_key_color),
                                  Text(separator, style=_separator_color),
                                  type_text))
    return Tree(type_text)


_indentation = 4


def _batch_str(batch_dims: BatchDimensions) -> str:
    if batch_dims is None:
        return ""
    assert len(batch_dims) == 1
    batch_dim = batch_dims[0]
    if batch_dim is None:
        return ""
    return f" batched over axis {batch_dim}"


def _indent_space(indent: int) -> str:
    return (indent * _indentation) * " "


@singledispatch
def _format_number(x: Any) -> str:
    return f"{x:10}"


@_format_number.register(np.inexact)
def _(x: np.inexact[Any]) -> str:
    return f"{x:10.4f}"


def _show_array(tree: Tree, array: NumpyArray) -> None:
    if not np.issubdtype(array.dtype, np.number) and not np.issubdtype(array.dtype, np.bool_):
        return
    if np.prod(array.shape) == 0:
        return
    if any(x > 12 for x in array.shape):
        tree.children.append(display_generic(np.mean(array), set(), key="mean"))
        tree.children.append(display_generic(np.std(array), set(), key="deviation"))
        return
    if 1 in array.shape:
        _show_array(tree,
                    array[tuple[Union[int, slice], ...](0 if s == 1 else slice(None)
                                                        for s in array.shape)])
        return
    if len(array.shape) == 0:
        tree.add(_format_number(array[()]))
        return
    if len(array.shape) > 2:
        return
    table = Table(show_header=False,
                  show_edge=False,
                  style=_table_color)
    for _ in range(array.shape[-1]):
        table.add_column()
    if len(array.shape) == 1:
        table.add_row(*(_format_number(array[i])
                        for i in range(array.shape[0])))
    elif len(array.shape) == 2:
        for j in range(array.shape[0]):
            table.add_row(*(_format_number(array[j, i])
                            for i in range(array.shape[1])))
    tree.add(table)


def _batch_dimension_iterator(values: Iterable[Any],
                              batch_dims: BatchDimensions = None) -> Iterable[BatchDimensions]:
    bdi = BatchDimensionIterator(batch_dims)
    for value in values:
        yield bdi.advance(value)
    bdi.check_done()
