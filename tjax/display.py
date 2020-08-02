import sys
from numbers import Number
from typing import Any, Type, Union

import colorful as cf
import networkx as nx
import numpy as np
from jax.interpreters.ad import JVPTracer
from jax.interpreters.partial_eval import JaxprTracer
from jax.interpreters.xla import DeviceArray

__all__ = ['Displayable', 'print_generic']


if sys.version_info >= (3, 8):
    from typing import Protocol, runtime_checkable

    @runtime_checkable
    class Displayable(Protocol):
        """
        This protocol identifies classes that support the `display_generic` mechanism.
        """
        def display(self, show_values: bool = True, indent: int = 0) -> str:
            ...
else:
    class Displayable:
        pass


def print_generic(*args: Any, **kwargs: Any) -> None:
    for value in args:
        print(display_generic(value))
    for key, value in kwargs.items():
        print(display_key_and_value(key, value, "=", True, 0))


def display_generic(value: Any, show_values: bool = True, indent: int = 0) -> str:
    # pylint: disable=too-many-return-statements
    if isinstance(value, JVPTracer):
        return cf.magenta(f"JVPTracer {value.shape} {value.dtype}") + "\n"
    if isinstance(value, JaxprTracer):
        return cf.magenta(f"JaxprTracer {value.shape} {value.dtype}") + "\n"
    if isinstance(value, (np.ndarray, DeviceArray)):
        if isinstance(value, np.ndarray):
            retval = cf.yellow(f"NumPy Array {value.shape}") + "\n"
        elif isinstance(value, DeviceArray):
            retval = cf.violet(f"Jax Array {value.shape}") + "\n"
        if show_values:
            retval += _show_array(indent + 1, value)
        return retval
    if isinstance(value, Number) or value is None:
        return cf.cyan(str(value)) + "\n"
    if (isinstance(value, Displayable)
            if sys.version_info >= (3, 8)
            else hasattr(value, 'display')):
        return value.display(show_values=show_values, indent=indent)
    if isinstance(value, dict):
        return (display_class(dict)
                + "".join(display_key_and_value(key, sub_value, "=", show_values, indent)
                          for key, sub_value in value.items()))
    if isinstance(value, (tuple, list)):
        return (display_class(type(value))
                + "".join(display_key_and_value("", sub_value, "", show_values, indent)
                          for sub_value in value))
    if isinstance(value, nx.Graph):
        return display_graph(value, show_values, indent)
    return cf.red(str(value)) + "\n"


def display_graph(graph: nx.Graph, show_values: bool, indent: int = 0) -> str:
    directed = isinstance(graph, nx.DiGraph)
    arrow = cf.base00('⟶  ' if directed else '🡘 ')
    retval = display_class(type(graph))
    for name, value in graph.nodes.items():
        retval += display_key_and_value(name, value['element'], ": ", show_values, indent)
    for (source, target), value in graph.edges.items():
        key = f"{source}{arrow}{target}"
        retval += display_key_and_value(key, value['link'], ": ", show_values, indent)
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
