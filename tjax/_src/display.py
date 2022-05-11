from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from functools import singledispatch
from numbers import Number
from typing import Any, Iterable, List, Optional, Set, Tuple, Type, TypeVar, Union

import colorful as cf
import flax.linen as nn
import numpy as np
from jax.errors import TracerArrayConversionError
from jax.experimental.host_callback import id_tap
from jax.interpreters.ad import JVPTracer
from jax.interpreters.batching import BatchTracer
from jax.interpreters.partial_eval import DynamicJaxprTracer, JaxprTracer
from jax.interpreters.xla import DeviceArray
from jax.tree_util import tree_leaves

from .annotations import Array, TapFunctionTransforms

__all__ = ['print_generic', 'display_generic', 'id_display']


# Redefinition typing errors in this file are due to https://github.com/python/mypy/issues/2904.


def print_generic(*args: Any,
                  batch_dims: Optional[Tuple[Optional[int], ...]] = None,
                  raise_on_nan: bool = True,
                  **kwargs: Any) -> None:
    bdi = BatchDimensionIterator(batch_dims)
    found_nan = False
    for value in args:
        sub_batch_dims = bdi.advance(value)
        s = display_generic(value, set(), batch_dims=sub_batch_dims)
        print(s)
        found_nan = found_nan or raise_on_nan and 'nan' in str(s)
    for key, value in kwargs.items():
        sub_batch_dims = bdi.advance(value)
        s = display_key_and_value(key, value, "=", set(), batch_dims=sub_batch_dims)
        print(s)
        found_nan = found_nan or raise_on_nan and 'nan' in str(s)
    if found_nan:
        assert False


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
def _(value: JVPTracer,
      seen: Set[int],
      show_values: bool = True,
      indent: int = 0,
      batch_dims: Optional[Tuple[Optional[int], ...]] = None) -> str:
    return cf.magenta(f"JVPTracer {value.shape} {value.dtype}") + "\n"


@display_generic.register
def _(value: JaxprTracer,
      seen: Set[int],
      show_values: bool = True,
      indent: int = 0,
      batch_dims: Optional[Tuple[Optional[int], ...]] = None) -> str:
    return cf.magenta(f"JaxprTracer {value.shape} {value.dtype}") + "\n"


@display_generic.register
def _(value: DynamicJaxprTracer,
      seen: Set[int],
      show_values: bool = True,
      indent: int = 0,
      batch_dims: Optional[Tuple[Optional[int], ...]] = None) -> str:
    return cf.magenta(f"DynamicJaxprTracer {value.shape} {value.dtype}") + "\n"


@display_generic.register
def _(value: BatchTracer,
      seen: Set[int],
      show_values: bool = True,
      indent: int = 0,
      batch_dims: Optional[Tuple[Optional[int], ...]] = None) -> str:
    return cf.magenta(f"BatchTracer {value.shape} {value.dtype} "
                      f"batched over {value.val.shape[value.batch_dim]}") + "\n"


@display_generic.register(np.ndarray)
def _(value: Array,
      seen: Set[int],
      show_values: bool = True,
      indent: int = 0,
      batch_dims: Optional[Tuple[Optional[int], ...]] = None) -> str:
    retval = cf.yellow(f"NumPy Array {value.shape} {value.dtype}{_batch_str(batch_dims)}") + "\n"
    return retval + _show_array(indent + 1, value)


@display_generic.register
def _(value: DeviceArray,
      seen: Set[int],
      show_values: bool = True,
      indent: int = 0,
      batch_dims: Optional[Tuple[Optional[int], ...]] = None) -> str:
    base_string = cf.violet(
        f"Jax Array {value.shape} {value.dtype}") + "\n"  # type: ignore[attr-defined]
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
        for sub_batch_dims, sub_value in zip(batch_dimension_iterator(value, batch_dims), value))


def display_dataclass(value: Any,
                      seen: Set[int],
                      show_values: bool = True,
                      indent: int = 0,
                      batch_dims: Optional[Tuple[Optional[int], ...]] = None) -> str:
    is_module = isinstance(value, nn.Module)  # pyright: ignore
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
            if not isinstance(child_module, nn.Module):  # pyright: ignore
                continue
            retval += display_key_and_value(name, child_module, "=", seen, show_values, indent + 1,
                                            None)
    return retval


_T = TypeVar('_T')


def id_display(x: _T, name: Optional[str] = None, *, no_jvp: bool = False) -> _T:
    def tap(x: _T, transforms: TapFunctionTransforms) -> None:
        nonlocal name
        batch_dims: Optional[Tuple[Optional[int], ...]] = None
        flags = []
        for transform_name, transform_dict in transforms:
            if transform_name == 'batch':
                batch_dims = transform_dict['batch_dims']
                continue
            if no_jvp and transform_name == 'jvp':
                return
            if transform_name in ['jvp', 'mask', 'transpose']:
                flags.append(transform_name)
                continue
        if name is None:
            print_generic(x, batch_dims=batch_dims)
        else:
            if flags:
                final_name = name + f" [{', '.join(flags)}]"
            else:
                final_name = name
            # https://github.com/python/mypy/issues/11583
            print_generic(batch_dims=batch_dims, **{final_name: x})  # type: ignore[arg-type]
    return id_tap(tap, x, result=x)  # type: ignore[no-untyped-call]


class BatchDimensionIterator:
    def __init__(self, batch_dims: Optional[Tuple[Optional[int], ...]] = None):
        super().__init__()
        self.batch_dims = batch_dims
        self.i = 0

    def advance(self, value: Any) -> Optional[Tuple[Optional[int], ...]]:
        if self.batch_dims is None:
            return None
        n = len(tree_leaves(value))
        old_i = self.i
        self.i += n
        return self.batch_dims[old_i: self.i]

    def check_done(self) -> None:
        if self.batch_dims is not None:
            assert self.i == len(self.batch_dims)


def batch_dimension_iterator(values: Iterable[Any],
                             batch_dims: Optional[Tuple[Optional[int], ...]] = None) -> Iterable[
                                 Optional[Tuple[Optional[int], ...]]]:
    bdi = BatchDimensionIterator(batch_dims)
    for value in values:
        yield bdi.advance(value)
    bdi.check_done()


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


def _show_array(indent: int, array: Array) -> str:
    if not np.issubdtype(array.dtype, np.number) and not np.issubdtype(array.dtype, np.bool_):
        return ""
    if np.prod(array.shape) == 0:
        return ""
    if any(x > 12 for x in array.shape):
        return ""
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
