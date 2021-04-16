from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple, Union

import numpy as np

__all__ = ['Shape', 'ShapeLike', 'SliceLike', 'Array', 'BooleanArray', 'IntegralArray', 'RealArray',
           'ComplexArray', 'Integral', 'Real', 'Complex', 'BooleanNumeric', 'IntegralNumeric',
           'RealNumeric', 'ComplexNumeric', 'PyTree', 'TapFunctionTransforms']

Shape = Tuple[int, ...]
ShapeLike = Union[int, Sequence[int]]
_SliceLikeItem = Union[int, None, slice]
SliceLike = Union[_SliceLikeItem, Tuple[_SliceLikeItem, ...]]


Array = np.ndarray
BooleanArray = Array
IntegralArray = Array
RealArray = Array
ComplexArray = Array


Integral = int  # Use this until numbers.Integral works with MyPy.
Real = Union[float, int]  # Use this until numbers.Real works with MyPy.
Complex = Union[complex, float, int]  # Use this until numbers.Complex works with MyPy.
BooleanNumeric = Union[BooleanArray, bool]
IntegralNumeric = Union[IntegralArray, Integral]
RealNumeric = Union[RealArray, Real]
ComplexNumeric = Union[ComplexArray, Complex]


PyTree = Any
TapFunctionTransforms = Sequence[Tuple[str, Dict[str, Any]]]