from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, MutableSet
from dataclasses import fields, is_dataclass
from functools import singledispatch
from numbers import Number
from typing import Any, Union

import numpy as np
from jax import Array
from jax.errors import TracerArrayConversionError
from jax.interpreters.batching import BatchTracer
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from ..annotations import NumpyArray
from ..dataclasses import DataclassInstance
from .batch_dimensions import BatchDimensionIterator, BatchDimensions
from .colors import solarized

__all__ = ['display_generic', 'display_class']


# Constants ----------------------------------------------------------------------------------------
# Numeric (warm)
_numpy_array_color = solarized['magenta']
_jax_array_color = solarized['red']
_number_color = solarized['yellow']
# solarized['orange']
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
FlaxModule: type[Any]
try:
    from flax.linen import Module as FlaxModule
    flax_loaded = True
except ImportError:
    flax_loaded = False
    FlaxModule = type(None)


@singledispatch
def display_generic(value: Any,
                    *,
                    seen: MutableSet[int],
                    show_values: bool = True,
                    key: str = '',
                    batch_dims: BatchDimensions | None = None) -> Tree:
    if (x := _verify(value, seen, key)) is not None:
        return x
    if is_dataclass(value) and not isinstance(value, type):
        return display_dataclass(value, seen=seen, show_values=show_values, key=key,
                                 batch_dims=batch_dims)
    return _assemble(key, Text(str(value), style=_unknown_color))


@display_generic.register
def _(value: str,
      *,
      seen: MutableSet[int],
      show_values: bool = True,
      key: str = '',
      batch_dims: tuple[int | None, ...] | None = None) -> Tree:
    if (x := _verify(value, seen, key)) is not None:
        return x
    return _assemble(key, Text(f'"{value}"', style=_string_color))


@display_generic.register(type)
def _(value: type[Any],
      *,
      seen: MutableSet[int],
      show_values: bool = True,
      key: str = '',
      batch_dims: BatchDimensions | None = None) -> Tree:
    if (x := _verify(value, seen, key)) is not None:
        return x
    return _assemble(key, Text(f"type[{value.__name__}]", style=_type_color))


@display_generic.register(np.ndarray)
def _(value: NumpyArray,
      *,
      seen: MutableSet[int],
      show_values: bool = True,
      key: str = '',
      batch_dims: BatchDimensions | None = None) -> Tree:
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
      seen: MutableSet[int],
      show_values: bool = True,
      key: str = '',
      batch_dims: BatchDimensions | None = None) -> Tree:
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
      seen: MutableSet[int],
      show_values: bool = True,
      key: str = '',
      batch_dims: BatchDimensions | None = None) -> Tree:
    if (x := _verify(value, seen, key)) is not None:
        return x
    return _assemble(key, Text(str(value), style=_number_color))


@display_generic.register(Mapping)
def _(value: Mapping[Any, Any],
      *,
      seen: MutableSet[int],
      show_values: bool = True,
      key: str = '',
      batch_dims: BatchDimensions | None = None) -> Tree:
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
      seen: MutableSet[int],
      show_values: bool = True,
      key: str = '',
      batch_dims: BatchDimensions | None = None) -> Tree:
    if (x := _verify(value, seen, key)) is not None:
        return x
    retval = display_class(key, type(value))
    for sub_batch_dims, sub_value in zip(_batch_dimension_iterator(value, batch_dims), value,
                                         strict=True):
        retval.children.append(display_generic(sub_value, seen=seen, show_values=show_values,
                                               key="", batch_dims=sub_batch_dims))
    return retval


def display_dataclass(value: DataclassInstance,
                      *,
                      seen: MutableSet[int],
                      show_values: bool = True,
                      key: str = '',
                      batch_dims: BatchDimensions | None = None) -> Tree:
    is_module = flax_loaded and isinstance(value, FlaxModule)
    retval = display_class(key, type(value), is_module=is_module)
    bdi = BatchDimensionIterator(batch_dims)
    for field_info in fields(value):
        name = field_info.name
        if is_module and name in {'parent', 'name'}:
            continue
        sub_value = getattr(value, name)
        sub_batch_dims = bdi.advance(sub_value)
        retval.children.append(display_generic(sub_value, seen=seen, show_values=show_values,
                                               key=name, batch_dims=sub_batch_dims))
    if is_module:
        assert isinstance(value, FlaxModule)
        retval.children.append(display_generic(value.name, seen=seen, show_values=show_values,
                                               key='name'))
        retval.children.append(display_generic(value.parent is not None, seen=seen,
                                               show_values=show_values, key='has_parent'))
        retval.children.append(display_generic(value.scope is not None, seen=seen,
                                               show_values=show_values, key='bound'))
        value_state = value._state  # pylint: disable=protected-access # noqa: SLF001
        for name, child_module in value_state.children.items():  # pytype: disable=attribute-error
            if not isinstance(child_module, FlaxModule):
                continue
            retval.children.append(display_generic(child_module, seen=seen, show_values=show_values,
                                                   key=name))
    return retval


# Public unexported functions ----------------------------------------------------------------------
def display_class(key: str, cls: type[Any], *, is_module: bool = False) -> Tree:
    type_color = _module_color if is_module else _class_color
    return _assemble(key, Text(cls.__name__, style=type_color))


def _verify(value: Any,
            seen: MutableSet[int],
            key: str) -> Tree | None:
    if id(value) in seen:
        return _assemble(key, Text('<seen>', style=_seen_color))
    if is_dataclass(value) and not isinstance(value, type):
        seen.add(id(value))
    return None


# Private functions --------------------------------------------------------------------------------
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
    if not issubclass(array.dtype.type, (np.bool_, np.number)):
        return
    if math.prod(array.shape) == 0:
        return
    if 1 in array.shape:
        _show_array(tree,
                    array[tuple[Union[int, slice], ...](0 if s == 1 else slice(None)
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
