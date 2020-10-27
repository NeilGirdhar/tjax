from __future__ import annotations

from typing import Any, Callable, Dict, Sequence, Tuple, TypeVar, Union

from chex import Array
from jax import vjp

__all__ = ['Shape',
           'ShapeLike',
           'SliceLike',
           'Array',
           'RealArray',
           'ComplexArray',
           'PyTree',
           'TapFunctionTransforms',
           'vjp_with_aux']


Shape = Tuple[int, ...]
ShapeLike = Union[int, Sequence[int]]
SliceLike = Tuple[Union[int, None, slice], ...]
RealArray = Array
ComplexArray = Array
PyTree = Any
TapFunctionTransforms = Sequence[Tuple[str, Dict[str, Any]]]


T = TypeVar('T')
U = TypeVar('U')


def vjp_with_aux(fun: Callable[..., Tuple[T, U]],
                 *primals: Any) -> Tuple[T, Callable[..., Any], U]:
    return vjp(fun, *primals, has_aux=True)  # type: ignore
