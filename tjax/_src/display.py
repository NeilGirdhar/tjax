from __future__ import annotations

from collections.abc import Mapping
from functools import singledispatch
from numbers import Number
from typing import Any, Iterable, List, Optional, Tuple, Type, TypeVar, Union

import colorful as cf
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
        s = display_generic(value, batch_dims=sub_batch_dims)
        print(s)
        found_nan = found_nan or raise_on_nan and 'nan' in str(s)
    for key, value in kwargs.items():
        sub_batch_dims = bdi.advance(value)
        s = display_key_and_value(key, value, "=", batch_dims=sub_batch_dims)
        print(s)
        found_nan = found_nan or raise_on_nan and 'nan' in str(s)
    if found_nan:
        assert False


@singledispatch
def display_generic(value: Any,
                    show_values: bool = True,
                    indent: int = 0,
                    batch_dims: Optional[Tuple[Optional[int], ...]] = None) -> str:
    return cf.red(str(value)) + "\n"


@display_generic.register
def _(value: JVPTracer,
      show_values: bool = True,
      indent: int = 0,
      batch_dims: Optional[Tuple[Optional[int], ...]] = None) -> str:
    return cf.magenta(f"JVPTracer {value.shape} {value.dtype}") + "\n"


@display_generic.register  # type: ignore
def _(value: JaxprTracer,
      show_values: bool = True,
      indent: int = 0,
      batch_dims: Optional[Tuple[Optional[int], ...]] = None) -> str:
    return cf.magenta(f"JaxprTracer {value.shape} {value.dtype}") + "\n"


@display_generic.register  # type: ignore
def _(value: DynamicJaxprTracer,
      show_values: bool = True,
      indent: int = 0,
      batch_dims: Optional[Tuple[Optional[int], ...]] = None) -> str:
    return cf.magenta(f"DynamicJaxprTracer {value.shape} {value.dtype}") + "\n"


@display_generic.register  # type: ignore
def _(value: BatchTracer,
      show_values: bool = True,
      indent: int = 0,
      batch_dims: Optional[Tuple[Optional[int], ...]] = None) -> str:
    return cf.magenta(f"BatchTracer {value.shape} {value.dtype} "
                      f"batched over {value.val.shape[value.batch_dim]}") + "\n"


@display_generic.register(np.ndarray)  # type: ignore
def _(value: Array,
      show_values: bool = True,
      indent: int = 0,
      batch_dims: Optional[Tuple[Optional[int], ...]] = None) -> str:
    retval = cf.yellow(f"NumPy Array {value.shape} {value.dtype}{_batch_str(batch_dims)}") + "\n"
    return retval + _show_array(indent + 1, value)


@display_generic.register  # type: ignore
def _(value: DeviceArray,
      show_values: bool = True,
      indent: int = 0,
      batch_dims: Optional[Tuple[Optional[int], ...]] = None) -> str:
    base_string = cf.violet(f"Jax Array {value.shape} {value.dtype}") + "\n"
    try:
        np_value = np.asarray(value)
    except TracerArrayConversionError:
        array_string = ""
    else:
        array_string = _show_array(indent + 1, np_value)
    return base_string + array_string


@display_generic.register(type(None))  # type: ignore
@display_generic.register(Number)
def _(value: Union[None, Number],
      show_values: bool = True,
      indent: int = 0,
      batch_dims: Optional[Tuple[Optional[int], ...]] = None) -> str:
    return cf.cyan(str(value)) + "\n"


@display_generic.register(Mapping)  # type: ignore
def _(value: Mapping[Any, Any],
      show_values: bool = True,
      indent: int = 0,
      batch_dims: Optional[Tuple[Optional[int], ...]] = None) -> str:
    return (display_class(type(value))
            + "".join(display_key_and_value(key, sub_value, "=", show_values, indent + 1)
                      for key, sub_value in sorted(value.items())))


@display_generic.register(tuple)  # type: ignore
@display_generic.register(list)
def _(value: Union[Tuple[Any, ...], List[Any]],
      show_values: bool = True,
      indent: int = 0,
      batch_dims: Optional[Tuple[Optional[int], ...]] = None) -> str:
    return display_class(type(value)) + "".join(
        display_key_and_value("", sub_value, "", show_values, indent + 1, sub_batch_dims)
        for sub_batch_dims, sub_value in zip(batch_dimension_iterator(value, batch_dims), value))


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
            print_generic(batch_dims=batch_dims, **{final_name: x})  # type: ignore
    return id_tap(tap, x, result=x)


class BatchDimensionIterator:
    def __init__(self, batch_dims: Optional[Tuple[Optional[int], ...]] = None):
        self.batch_dims = batch_dims
        self.i = 0

    def advance(self, value: Any) -> Optional[Tuple[Optional[int], ...]]:
        if self.batch_dims is None:
            return None
        n = len(tree_leaves(value))  # type: ignore
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
def display_class(cls: Type[Any]) -> str:
    return cf.orange(cls.__name__) + "\n"


def display_key_and_value(key: str,
                          value: Any,
                          separator: str,
                          show_values: bool = True,
                          indent: int = 0,
                          batch_dims: Optional[Tuple[Optional[int], ...]] = None) -> str:
    return (_indent_space(indent) + cf.blue(key) + cf.base00(separator)
            + display_generic(value, show_values, indent, batch_dims))


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


def _format_number(x: float) -> str:
    return f"{x:10.4f}"


def _show_array(indent: int, array: Array) -> str:
    if len(array.shape) == 0:
        return _indent_space(indent) + _format_number(array[()]) + "\n"
    if np.prod(array.shape) == 0:
        return ""
    if 1 in array.shape:
        return _show_array(indent, array[tuple(0 if s == 1 else slice(None)
                                               for s in array.shape)])
    if any(x > 12 for x in array.shape):
        return ""
    if len(array.shape) == 1:
        return _indent_space(indent) + "  ".join(_format_number(array[i])
                                                 for i in range(array.shape[0])) + "\n"
    if len(array.shape) == 2:
        return "".join(_show_array(indent, array_slice)
                       for array_slice in array)
    return ""
