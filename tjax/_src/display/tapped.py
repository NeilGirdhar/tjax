from __future__ import annotations

from typing import Any, Optional, Tuple, TypeVar, Union, overload

from jax.experimental.host_callback import id_tap

from ..annotations import TapFunctionTransforms
from .print_generic import print_generic

__all__ = ['tapped_print_generic']

_T = TypeVar('_T')
_U = TypeVar('_U')


@overload
def tapped_print_generic(x: _T,
                         name: Optional[str] = None,
                         *,
                         no_jvp: bool = False,
                         result: None = None) -> _T:
    ...


@overload
def tapped_print_generic(x: Any,
                         name: Optional[str] = None,
                         *,
                         no_jvp: bool = False,
                         result: _U) -> _U:
    ...


def tapped_print_generic(x: _T,
                         name: Optional[str] = None,
                         *,
                         no_jvp: bool = False,
                         result: Optional[_U] = None) -> Union[_T, _U]:
    """
    Uses print_generic in a tapped function.
    """
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
    return id_tap(tap, x, result=x if result is None else result)  # type: ignore[no-untyped-call]
