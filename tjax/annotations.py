from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple, Union

from chex import Array

__all__ = ['Shape',
           'ShapeLike',
           'SliceLike',
           'Array',
           'BoolArray',
           'IntegerArray',
           'RealArray',
           'ComplexArray',
           'PyTree',
           'TapFunctionTransforms']


Shape = Tuple[int, ...]
ShapeLike = Union[int, Sequence[int]]
SliceLike = Tuple[Union[int, None, slice], ...]
BoolArray = Array
IntegerArray = Array
RealArray = Array
ComplexArray = Array
PyTree = Any
TapFunctionTransforms = Sequence[Tuple[str, Dict[str, Any]]]
