from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from functools import singledispatch
from numbers import Number
from typing import Any, Iterable, List, Optional, Set, Tuple, Type, Union

import colorful as cf
import numpy as np
from jax import Array
from jax.errors import TracerArrayConversionError
from jax.interpreters.batching import BatchTracer

from ..annotations import NumpyArray
from .batch_dimensions import BatchDimensionIterator

__all__ = ['display_generic', 'display_class', 'display_key_and_value']


_unknown_color = 'red'
_type_color = 'red'
_batch_array_color = 'magenta'
_numpy_array_color = 'yellow'
_jax_array_color = 'violet'
_number_color = 'cyan'
_flax_module_color = 'green'
_class_color = 'orange'
_key_color = 'blue'
_separator_color = 'base00'


FlaxModule: Type[Any]
try:
    from flax.linen import Module as FlaxModule
    flax_loaded = True
except ImportError:
    flax_loaded = False
    FlaxModule = type(None)


@singledispatch
def display_generic(value: Any,
                    seen: Set[int],
                    show_values: bool = True,
                    indent: int = 0,
                    batch_dims: Optional[Tuple[Optional[int], ...]] = None) -> str:
    if is_dataclass(value) and not isinstance(value, type):
        return display_dataclass(value, seen, show_values, indent, batch_dims)
    return cf.red(str(value)) + "\n"


@display_generic.register(type)
def _(value: Type[Any],
      seen: Set[int],
      show_values: bool = True,
      indent: int = 0,
      batch_dims: Optional[Tuple[Optional[int], ...]] = None) -> str:
    return cf.red(f"type[{value.__name__}]") + "\n"


@display_generic.register
def _(value: BatchTracer,
      seen: Set[int],
      show_values: bool = True,
      indent: int = 0,
      batch_dims: Optional[Tuple[Optional[int], ...]] = None) -> str:
    return cf.magenta(f"BatchTracer {value.shape} {value.dtype} "
                      f"batched over {value.val.shape[value.batch_dim]}") + "\n"


@display_generic.register(np.ndarray)
def _(value: NumpyArray,
      seen: Set[int],
      show_values: bool = True,
      indent: int = 0,
      batch_dims: Optional[Tuple[Optional[int], ...]] = None) -> str:
    retval = cf.yellow(f"NumPy Array {value.shape} {value.dtype}{_batch_str(batch_dims)}") + "\n"
    return retval + _show_array(indent + 1, value)


@display_generic.register
def _(value: Array,
      seen: Set[int],
      show_values: bool = True,
      indent: int = 0,
      batch_dims: Optional[Tuple[Optional[int], ...]] = None) -> str:
    base_string = cf.violet(
        f"Jax Array {value.shape} {value.dtype}") + "\n"
    try:
        np_value = np.asarray(value)
    except TracerArrayConversionError:
        array_string = ""
    else:
        array_string = _show_array(indent + 1, np_value)
    return base_string + array_string


@display_generic.register(type(None))
@display_generic.register(Number)
def _(value: Union[None, Number],
      seen: Set[int],
      show_values: bool = True,
      indent: int = 0,
      batch_dims: Optional[Tuple[Optional[int], ...]] = None) -> str:
    return cf.cyan(str(value)) + "\n"


@display_generic.register(Mapping)
def _(value: Mapping[Any, Any],
      seen: Set[int],
      show_values: bool = True,
      indent: int = 0,
      batch_dims: Optional[Tuple[Optional[int], ...]] = None) -> str:
    return (display_class(type(value))
            + "".join(display_key_and_value(key, sub_value, "=", seen, show_values, indent + 1)
                      for key, sub_value in sorted(value.items())))


@display_generic.register(tuple)
@display_generic.register(list)
def _(value: Union[Tuple[Any, ...], List[Any]],
      seen: Set[int],
      show_values: bool = True,
      indent: int = 0,
      batch_dims: Optional[Tuple[Optional[int], ...]] = None) -> str:
    return display_class(type(value)) + "".join(
        display_key_and_value("", sub_value, "", seen, show_values, indent + 1, sub_batch_dims)
        for sub_batch_dims, sub_value in zip(_batch_dimension_iterator(value, batch_dims), value))


def display_dataclass(value: Any,
                      seen: Set[int],
                      show_values: bool = True,
                      indent: int = 0,
                      batch_dims: Optional[Tuple[Optional[int], ...]] = None) -> str:
    is_module = flax_loaded and isinstance(value, FlaxModule)
    retval = display_class(type(value), is_module)
    bdi = BatchDimensionIterator(batch_dims)
    for field_info in fields(value):
        name = field_info.name
        if is_module and name in {'parent', 'name'}:
            continue
        sub_value = getattr(value, name)
        sub_batch_dims = bdi.advance(sub_value)
        retval += display_key_and_value(name, sub_value, "=", seen, show_values, indent + 1,
                                        sub_batch_dims)
    if is_module:
        retval += display_key_and_value('name', value.name, "=", seen, show_values, indent + 1,
                                        None)
        retval += display_key_and_value('has_parent', value.parent is not None, "=", seen,
                                        show_values, indent + 1, None)
        retval += display_key_and_value('bound', value.scope is not None, "=", seen, show_values,
                                        indent + 1, None)
        # pylint: disable=protected-access
        for name, child_module in value._state.children.items():  # pytype: disable=attribute-error
            if not isinstance(child_module, FlaxModule):
                continue
            retval += display_key_and_value(name, child_module, "=", seen, show_values, indent + 1,
                                            None)
    return retval


# Public unexported functions ----------------------------------------------------------------------
def display_class(cls: Type[Any], is_module: bool = False) -> str:
    color_f = cf.green if is_module else cf.orange
    return color_f(cls.__name__) + "\n"


def display_key_and_value(key: str,
                          value: Any,
                          separator: str,
                          seen: Set[int],
                          show_values: bool = True,
                          indent: int = 0,
                          batch_dims: Optional[Tuple[Optional[int], ...]] = None) -> str:
    if id(value) in seen:
        value = "<seen>"
    elif is_dataclass(value) and not isinstance(value, type):
        seen.add(id(value))
    return (_indent_space(indent) + cf.blue(key) + cf.base00(separator)
            + display_generic(value, seen, show_values, indent, batch_dims))


# Private functions --------------------------------------------------------------------------------
_indentation = 4


def _batch_str(batch_dims: Optional[Tuple[Optional[int], ...]]) -> str:
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


def _show_array(indent: int, array: NumpyArray) -> str:
    if not np.issubdtype(array.dtype, np.number) and not np.issubdtype(array.dtype, np.bool_):
        return ""
    if np.prod(array.shape) == 0:
        return ""
    if any(x > 12 for x in array.shape):
        return (_indent_space(indent)
                + f"mean {np.mean(array):10.4f}; deviation {np.std(array):10.4f}\n")
    if 1 in array.shape:
        return _show_array(indent, array[tuple[Union[int, slice], ...](0 if s == 1 else slice(None)
                                                                       for s in array.shape)])
    if len(array.shape) == 0:
        return _indent_space(indent) + _format_number(array[()]) + "\n"
    if len(array.shape) == 1:
        return _indent_space(indent) + "  ".join(_format_number(array[i])
                                                 for i in range(array.shape[0])) + "\n"
    if len(array.shape) == 2:
        return "".join(_show_array(indent, array_slice)
                       for array_slice in array)
    return ""


def _batch_dimension_iterator(values: Iterable[Any],
                              batch_dims: Optional[Tuple[Optional[int], ...]] = None) -> Iterable[
                                  Optional[Tuple[Optional[int], ...]]]:
    bdi = BatchDimensionIterator(batch_dims)
    for value in values:
        yield bdi.advance(value)
    bdi.check_done()
