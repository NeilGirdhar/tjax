from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, TypeVar, overload

from jax.experimental.host_callback import id_tap
from rich.console import Console

from ..annotations import TapFunctionTransforms
from .print_generic import print_generic

__all__ = ['tapped_print_generic']

_T = TypeVar('_T')
_U = TypeVar('_U')


@overload
def tapped_print_generic(*args: Any,
                         raise_on_nan: bool = True,
                         console: Optional[Console] = None,
                         no_jvp: bool = False,
                         result: None = None,
                         **kwargs: Any
                         ) -> Any:
    ...


@overload
def tapped_print_generic(*args: Any,
                         raise_on_nan: bool = True,
                         console: Optional[Console] = None,
                         no_jvp: bool = False,
                         result: _U,
                         **kwargs: Any
                         ) -> _U:
    ...


def tapped_print_generic(*args: Any,
                         raise_on_nan: bool = True,
                         console: Optional[Console] = None,
                         no_jvp: bool = False,
                         result: Any = None,
                         **kwargs: Any
                         ) -> Any:
    """
    Uses print_generic in a tapped function.
    """
    def tap(x: Tuple[Tuple[Any, ...], Dict[str, Any]],
            transforms: TapFunctionTransforms
            ) -> None:
        args, kwargs = x
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
            result = next(iter(kwargs))
        else:
            assert False
    return id_tap(tap, (args, kwargs), result=result)  # type: ignore[no-untyped-call]