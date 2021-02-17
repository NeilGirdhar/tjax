from __future__ import annotations

from typing import Any, Sequence, Tuple, Union

from chex import Array

__all__ = ['Shape',
           'ShapeLike',
           'SliceLike',
           'Array',
           'RealArray',
           'ComplexArray',
           'PyTree']


Shape = Tuple[int, ...]
ShapeLike = Union[int, Sequence[int]]
SliceLike = Tuple[Union[int, None, slice], ...]
RealArray = Array
ComplexArray = Array
PyTree = Any
