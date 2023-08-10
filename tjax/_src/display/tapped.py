from __future__ import annotations

from typing import Any, TypeVar, overload

from jax.experimental.host_callback import id_tap
from jax.tree_util import tree_flatten, tree_unflatten
from rich.console import Console

from ..annotations import TapFunctionTransforms
from .batch_dimensions import BatchDimensions, combine_batch_dimensions
from .print_generic import print_generic

__all__ = ['tapped_print_generic']

_U = TypeVar('_U')


@overload
def tapped_print_generic(*args: Any,
                         raise_on_nan: bool = True,
                         console: Console | None = None,
                         no_jvp: bool = False,
                         result: None = None,
                         **kwargs: Any
                         ) -> Any:
    ...


@overload
def tapped_print_generic(*args: Any,
                         raise_on_nan: bool = True,
                         console: Console | None = None,
                         no_jvp: bool = False,
                         result: _U,
                         **kwargs: Any
                         ) -> _U:
    ...


def tapped_print_generic(*args: Any,
                         raise_on_nan: bool = True,
                         console: Console | None = None,
                         no_jvp: bool = False,
                         result: Any = None,
                         **kwargs: Any
                         ) -> Any:
    """Uses print_generic in a tapped function.

    Args:
        raise_on_nan: Assert if NaN is found in the output.
        console: The console that formats the output.
        no_jvp: Stifle printout of JVP tangents.
        result: A tracer to be returned to ensure sequencing.
        args: Positional arguments to be printed.  Only dynamic arguments are allowed.
        kwargs: Keyword arguments to be printed.  Only static keys and dynamic values are allowed.
    Returns: The value of result, or else the lone element of args and kwargs.
    """
    leaves, tree_def = tree_flatten((args, kwargs))

    def tap(tap_leaves: list[Any],
            transforms: TapFunctionTransforms
            ) -> None:
        args, kwargs = tree_unflatten(tree_def, tap_leaves)
        batch_dims: BatchDimensions = None
        flags = []
        for transform_name, transform_dict in transforms:
            if transform_name == 'batch':
                batch_dims = combine_batch_dimensions(batch_dims, transform_dict['batch_dims'])
                continue
            if no_jvp and transform_name == 'jvp':
                return
            if transform_name in ['jvp', 'mask', 'transpose']:
                flags.append(transform_name)
                continue
        modified_kwargs = ({key + f" [{', '.join(flags)}]": value
                            for key, value in kwargs.items()}
                           if flags
                           else kwargs)
        print_generic(*args, batch_dims=batch_dims, raise_on_nan=raise_on_nan, console=console,
                      **modified_kwargs)

    if result is None:
        if args:
            assert len(args) == 1
            result = args[0]
        elif kwargs:
            assert len(kwargs) == 1
            result = next(iter(kwargs.values()))
        else:
            assert False
    return id_tap(tap, leaves, result=result)  # type: ignore[no-untyped-call]
