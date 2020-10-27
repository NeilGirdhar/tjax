from functools import singledispatch
from numbers import Number
from typing import Any, Dict, List, Tuple, Type, Union

import colorful as cf
import networkx as nx
import numpy as np
from jax.interpreters.ad import JVPTracer
from jax.interpreters.batching import BatchTracer
from jax.interpreters.partial_eval import JaxprTracer
from jax.interpreters.xla import DeviceArray

__all__ = ['print_generic', 'display_generic']


# Redefinition typing errors in this file are due to https://github.com/python/mypy/issues/2904.


def print_generic(*args: Any, **kwargs: Any) -> None:
    for value in args:
        print(display_generic(value))
    for key, value in kwargs.items():
        print(display_key_and_value(key, value, "=", True, 0))


@singledispatch
def display_generic(value: Any, show_values: bool = True, indent: int = 0) -> str:
    return cf.red(str(value)) + "\n"


@display_generic.register
def _(value: JVPTracer, show_values: bool = True, indent: int = 0) -> str:
    return cf.magenta(f"JVPTracer {value.shape} {value.dtype}") + "\n"


@display_generic.register  # type: ignore
def _(value: JaxprTracer, show_values: bool = True, indent: int = 0) -> str:
    return cf.magenta(f"JaxprTracer {value.shape} {value.dtype}") + "\n"


@display_generic.register  # type: ignore
def _(value: BatchTracer, show_values: bool = True, indent: int = 0) -> str:
    return cf.magenta(f"BatchTracer {value.shape} {value.dtype} "
                      f"batched over {value.val.shape[value.batch_dim]}") + "\n"


@display_generic.register  # type: ignore
def _(value: np.ndarray, show_values: bool = True, indent: int = 0) -> str:
    retval = cf.yellow(f"NumPy Array {value.shape}") + "\n"
    return retval + _show_array(indent + 1, value)


@display_generic.register  # type: ignore
def _(value: DeviceArray, show_values: bool = True, indent: int = 0) -> str:
    retval = cf.violet(f"Jax Array {value.shape}") + "\n"
    return retval + _show_array(indent + 1, value)


@display_generic.register(type(None))  # type: ignore
@display_generic.register(Number)
def _(value: Union[None, Number], show_values: bool = True, indent: int = 0) -> str:
    return cf.cyan(str(value)) + "\n"


@display_generic.register(dict)  # type: ignore
def _(value: Dict[Any, Any], show_values: bool = True, indent: int = 0) -> str:
    return (display_class(dict)
            + "".join(display_key_and_value(key, sub_value, "=", show_values, indent)
                      for key, sub_value in sorted(value.items())))


@display_generic.register(tuple)  # type: ignore
@display_generic.register(list)
def _(value: Union[Tuple[Any, ...], List[Any]], show_values: bool = True, indent: int = 0) -> str:
    return (display_class(type(value))
            + "".join(display_key_and_value("", sub_value, "", show_values, indent)
                      for sub_value in value))


@display_generic.register  # type: ignore
def _(value: nx.Graph, show_values: bool, indent: int = 0) -> str:
    directed = isinstance(value, nx.DiGraph)
    arrow = cf.base00('⟶  ' if directed else '🡘 ')
    retval = display_class(type(value))
    for name, node in value.nodes.items():
        retval += display_key_and_value(name, node, ": ", show_values, indent)
    for (source, target), edge in value.edges.items():
        key = f"{source}{arrow}{target}"
        retval += display_key_and_value(key, edge, ": ", show_values, indent)
    return retval


def display_class(cls: Type[Any]) -> str:
    return cf.orange(cls.__name__) + "\n"


def display_key_and_value(
        key: str, value: Any, separator: str, show_values: bool = True, indent: int = 0) -> str:
    return (_indent_space(indent + 1) + cf.blue(key) + cf.base00(separator)
            + display_generic(value, show_values, indent + 1))


# Private functions --------------------------------------------------------------------------------
_indentation = 4


def _indent_space(indent: int) -> str:
    return (indent * _indentation) * " "


def _format_number(x: float) -> str:
    return f"{x:10.4f}"


def _show_array(indent: int, array: Union[np.ndarray, DeviceArray]) -> str:
    if len(array.shape) == 0:
        return _indent_space(indent) + _format_number(array[()]) + "\n"
    if np.prod(array.shape) == 0:
        return ""
    if 1 in array.shape:
        return _show_array(indent, array[tuple(0 if s == 1 else slice(None)
                                               for s in array.shape)])
    if len(array.shape) == 1:
        return _indent_space(indent) + "  ".join(_format_number(array[i])
                                                 for i in range(array.shape[0])) + "\n"
    if len(array.shape) == 2:
        return "".join(_show_array(indent, array_slice)
                       for array_slice in array)
    return ""
