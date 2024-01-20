from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, MutableSet
from dataclasses import fields, is_dataclass
from functools import singledispatch
from numbers import Number
from types import FunctionType
from typing import Any

import numpy as np
from jax import Array
from jax.errors import TracerArrayConversionError
from jax.interpreters.batching import BatchTracer
from jax.tree_util import PyTreeDef
from jaxlib.xla_extension import PjitFunction
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from tjax.dataclasses import DataclassInstance

from ..annotations import NumpyArray
from .batch_dimensions import BatchDimensionIterator, BatchDimensions
from .colors import solarized

__all__ = ['display_class', 'display_generic']


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
_unknown_color = f"{solarized['red']} bold"
_seen_color = solarized['red']


# Extra imports ------------------------------------------------------------------------------------
FlaxModule: type[Any]
FlaxVariable: type[Any]
try:
    from flax.experimental import nnx
except ImportError:
    flax_loaded = False
    def is_node_type(x: type[Any]) -> bool:
        return False
    FlaxModule = type(None)
    FlaxVariable = type(None)
else:
    flax_loaded = True
    FlaxModule = nnx.Module
    FlaxVariable = nnx.Variable
    is_node_type = nnx.graph_utils.is_node_type


@singledispatch
def display_generic(value: Any,
                    *,
                    seen: MutableSet[int] | None = None,
                    show_values: bool = True,
                    key: str = '',
                    batch_dims: BatchDimensions | None = None) -> Tree:
    if seen is None:
        seen = set()
    if (x := _verify(value, seen, key)) is not None:
        return x
    if is_dataclass(value) and not isinstance(value, type):
        return _display_dataclass(value, seen=seen, show_values=show_values, key=key,
                                  batch_dims=batch_dims)
    return _display_object(value, seen=seen, show_values=show_values, key=key,
                           batch_dims=batch_dims)
    # _assemble(key, Text(str(value), style=_unknown_color))


@display_generic.register
def _(value: str,
      *,
      seen: MutableSet[int] | None = None,
      show_values: bool = True,
      key: str = '',
      batch_dims: tuple[int | None, ...] | None = None) -> Tree:
    if seen is None:
        seen = set()
    if (x := _verify(value, seen, key)) is not None:
        return x
    return _assemble(key, Text(f'"{value}"', style=_string_color))


@display_generic.register(type)
def _(value: type[Any],
      *,
      seen: MutableSet[int] | None = None,
      show_values: bool = True,
      key: str = '',
      batch_dims: BatchDimensions | None = None) -> Tree:
    if seen is None:
        seen = set()
    if (x := _verify(value, seen, key)) is not None:
        return x
    return _assemble(key, Text(f"type[{value.__name__}]", style=_type_color))


@display_generic.register(np.ndarray)
def _(value: NumpyArray,
      *,
      seen: MutableSet[int] | None = None,
      show_values: bool = True,
      key: str = '',
      batch_dims: BatchDimensions | None = None) -> Tree:
    if seen is None:
        seen = set()
    if (x := _verify(value, seen, key)) is not None:
        return x
    extracted_batch_sizes, shape = _batched_axis_sizes_from_array_and_dims(value, batch_dims)
    batch_str = _batch_str(extracted_batch_sizes)
    retval = _assemble(key,
                       Text(f"NumPy Array {shape} {value.dtype}{batch_str}",
                            style=_numpy_array_color))
    _show_array(retval, value)
    return retval


@display_generic.register
def _(value: Array,
      *,
      seen: MutableSet[int] | None = None,
      show_values: bool = True,
      key: str = '',
      batch_dims: BatchDimensions | None = None) -> Tree:
    if seen is None:
        seen = set()
    if (x := _verify(value, seen, key)) is not None:
        return x
    # extracted_batch_sizes = _batched_axis_sizes_from_jax_array(value)
    extracted_batch_sizes = ()
    batch_str = _batch_str(extracted_batch_sizes)
    retval = _assemble(key,
                       Text(f"Jax Array {value.shape} {value.dtype}{batch_str}",
                            style=_jax_array_color))
    try:
        np_value = np.asarray(value)
    except TracerArrayConversionError:
        pass
    else:
        _show_array(retval, np_value)
    return retval


@display_generic.register(type(None))
@display_generic.register(Number)
def _(value: None | Number,
      *,
      seen: MutableSet[int] | None = None,
      show_values: bool = True,
      key: str = '',
      batch_dims: BatchDimensions | None = None) -> Tree:
    if seen is None:
        seen = set()
    if (x := _verify(value, seen, key)) is not None:
        return x
    return _assemble(key, Text(str(value), style=_number_color))


@display_generic.register(Mapping)
def _(value: Mapping[Any, Any],
      *,
      seen: MutableSet[int] | None = None,
      show_values: bool = True,
      key: str = '',
      batch_dims: BatchDimensions | None = None) -> Tree:
    if seen is None:
        seen = set()
    if (x := _verify(value, seen, key)) is not None:
        return x
    retval = display_class(key, type(value))
    for sub_batch_dims, (sub_key, sub_value) in zip(_batch_dimension_iterator(value.values(),
                                                                              batch_dims),
                                                    value.items(),
                                                    strict=True):
        retval.children.append(display_generic(sub_value, seen=seen, show_values=show_values,
                                               key=str(sub_key), batch_dims=sub_batch_dims))
    return retval


@display_generic.register(tuple)
@display_generic.register(list)
def _(value: tuple[Any, ...] | list[Any],
      *,
      seen: MutableSet[int] | None = None,
      show_values: bool = True,
      key: str = '',
      batch_dims: BatchDimensions | None = None) -> Tree:
    if seen is None:
        seen = set()
    if (x := _verify(value, seen, key)) is not None:
        return x
    retval = display_class(key, type(value))
    for sub_batch_dims, sub_value in zip(_batch_dimension_iterator(value, batch_dims), value,
                                         strict=True):
        retval.children.append(display_generic(sub_value, seen=seen, show_values=show_values,
                                               key="", batch_dims=sub_batch_dims))
    return retval


@display_generic.register(PyTreeDef)
def _(value: PyTreeDef,
      *,
      seen: MutableSet[int] | None = None,
      show_values: bool = True,
      key: str = '',
      batch_dims: BatchDimensions | None = None) -> Tree:
    if seen is None:
        seen = set()
    if (x := _verify(value, seen, key)) is not None:
        return x
    retval = display_class(key, type(value))
    retval.children.append(display_generic(hash(value), seen=seen, show_values=show_values,
                                           key="hash"))
    return retval


@display_generic.register(FunctionType)
@display_generic.register(PjitFunction)
def _(value: PyTreeDef,
      *,
      seen: MutableSet[int] | None = None,
      show_values: bool = True,
      key: str = '',
      batch_dims: BatchDimensions | None = None) -> Tree:
    if seen is None:
        seen = set()
    if (x := _verify(value, seen, key)) is not None:
        return x
    name = getattr(value, '__qualname__', "")
    retval = display_class(key, type(value))
    retval.children.append(display_generic(name, seen=seen, show_values=show_values, key="name"))
    return retval


if flax_loaded:
    @display_generic.register(FlaxVariable)
    def _(value: nnx.Variable[Any],
          *,
          seen: MutableSet[int] | None = None,
          show_values: bool = True,
          key: str = '',
          batch_dims: BatchDimensions | None = None) -> Tree:
        if seen is None:
            seen = set()
        if (x := _verify(value, seen, key)) is not None:
            return x
        retval = display_class(key, type(value))
        variables = _variables(value)
        variables = {key: sub_value
                     for key, sub_value in variables.items()
                     if not (key.endswith('_hooks') and value)}
        for name, sub_value in variables.items():
            retval.children.append(display_generic(sub_value, seen=seen, show_values=show_values,
                                                   key=name, batch_dims=None))
        return retval


# Public unexported functions ----------------------------------------------------------------------
def display_class(key: str, cls: type[Any]) -> Tree:
    name = cls.__name__
    tags = []
    if is_dataclass(cls):
        tags.append('dataclass')
    if flax_loaded and cls not in {int, float, tuple, list, set, dict}:
        if issubclass(cls, FlaxModule):
            tags.append('flax-module')
        elif is_node_type(cls):
            tags.append('flax-node')
    if tags:
        name += f"[{','.join(tags)}]"
    return _assemble(key, Text(name, style=_class_color))


# Private functions --------------------------------------------------------------------------------
def _display_dataclass(value: DataclassInstance,
                       *,
                       seen: MutableSet[int] | None = None,
                       show_values: bool = True,
                       key: str = '',
                       batch_dims: BatchDimensions | None = None) -> Tree:
    if seen is None:
        seen = set()
    retval = display_class(key, type(value))
    bdi = BatchDimensionIterator(batch_dims)
    names = set()
    for field_info in fields(value):
        name = field_info.name
        names.add(name)
        display_name = name
        if not field_info.init:
            display_name += ' (module)'
        sub_value = getattr(value, name, None)
        sub_batch_dims = (bdi.advance(sub_value)
                          if field_info.metadata.get('pytree_node', True)
                          else None)
        retval.children.append(display_generic(sub_value, seen=seen, show_values=show_values,
                                               key=display_name, batch_dims=sub_batch_dims))
    variables = _variables(value)
    for name, sub_value in variables.items():
        if name in names:
            continue
        retval.children.append(display_generic(sub_value, seen=seen, show_values=show_values,
                                               key=name + '*', batch_dims=None))
    return retval


def _display_object(value: Any,
                    *,
                    seen: MutableSet[int],
                    show_values: bool = True,
                    key: str = '',
                    batch_dims: BatchDimensions | None = None) -> Tree:
    retval = display_class(key, type(value))
    variables = _variables(value)
    for name, sub_value in variables.items():
        retval.children.append(display_generic(sub_value, seen=seen, show_values=show_values,
                                               key=name, batch_dims=None))
    return retval


def _variables(value: Any) -> dict[str, Any]:
    try:
        variables = vars(value)
    except TypeError:
        variables = ({name: getattr(value, name) for name in value.__slots__}
                     if hasattr(value, '__slots__')
                     else {})
    return {key: value
            for key, value in variables.items()
            if key != '_module__state'
            if not key.startswith('__')}


def _verify(value: Any,
            seen: MutableSet[int],
            key: str) -> Tree | None:
    if id(value) in seen:
        return _assemble(key, Text('<seen>', style=_seen_color))
    if is_dataclass(value) and not isinstance(value, type):
        seen.add(id(value))
    return None


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
def _format_number(x: Any) -> str:
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
        tree.children.append(display_generic(float(np.mean(xarray)), seen=set(), key="mean"))
        tree.children.append(display_generic(float(np.std(xarray)), seen=set(), key="deviation"))
        return
    if len(array.shape) == 0:
        tree.add(_format_number(array[()]))
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
    tree.add(table)


def _batch_dimension_iterator(values: Iterable[Any],
                              batch_dims: BatchDimensions | None = None
                              ) -> Iterable[BatchDimensions]:
    """Traverse values and batch_dims in parallel.

    Returns: An iterable of BatchDimensions objects for sub-elements of value.
    """
    bdi = BatchDimensionIterator(batch_dims)
    for value in values:
        yield bdi.advance(value)
    bdi.check_done()


def _batch_str(extracted_batch_sizes: tuple[int, ...]) -> str:
    return (f" batched over axes of size {extracted_batch_sizes}"
            if extracted_batch_sizes
            else "")


def _batched_axis_sizes_from_array_and_dims(value: NumpyArray,
                                            batch_dims: BatchDimensions
                                            ) -> tuple[tuple[int, ...],
                                                       tuple[int, ...]]:
    """Discover the shapes of the batched axes and the shape of the array without them.

    This function uses batch dimension information and a NumPy array, so it is only effective on
    jax.Arrays in tapped display.

    Returns: A tuple:
        The size of the axes that are batched over for a given Numpy array.
        The shape of the Numpy array without the batched axes.
    """
    shape = value.shape
    if batch_dims is None:
        return (), shape
    assert len(batch_dims) == 1
    batch_dim_tuple = batch_dims[0]
    if batch_dim_tuple is None:
        return (), shape
    assert isinstance(batch_dim_tuple, tuple)

    batch_sizes = []
    for batch_dim in reversed(batch_dim_tuple):
        batch_sizes.append(shape[batch_dim])
        shape = shape[: batch_dim] + shape[batch_dim + 1:]
    return tuple(batch_sizes), shape


def _batched_axis_sizes_from_jax_array(value: Array) -> tuple[int, ...]:
    """Discover the shapes of the batched axes.

    This function works by looking for BatchTracer objects, so it is only effective on jax.Arrays in
    non-tapped display.  This no longer works.

    Returns: The size of the axes that are batched over for a given Jax array.
    """
    batch_sizes = []
    while isinstance(value, BatchTracer):
        batch_sizes.append(value.val.shape[value.batch_dim])
        value = value.val
    return tuple(reversed(batch_sizes))
