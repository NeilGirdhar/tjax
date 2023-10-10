from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from jax import Array as JaxArray

__all__ = ['Shape', 'ShapeLike', 'SliceLike', 'Array', 'BooleanArray', 'IntegralArray', 'RealArray',
           'ComplexArray', 'Integral', 'Real', 'Complex', 'BooleanNumeric', 'IntegralNumeric',
           'RealNumeric', 'ComplexNumeric', 'PyTree', 'TapFunctionTransforms', 'NumpyArray',
           'NumpyBooleanArray', 'NumpyIntegralArray', 'NumpyRealArray', 'NumpyComplexArray',
           'NumpyBooleanNumeric', 'NumpyIntegralNumeric', 'NumpyRealNumeric', 'NumpyComplexNumeric',
           'JaxArray', 'JaxBooleanArray', 'JaxIntegralArray', 'JaxRealArray', 'JaxComplexArray',
           'KeyArray']

Shape = tuple[int, ...]
ShapeLike = int | Sequence[int]
_SliceLikeItem = int | None | slice
SliceLike = _SliceLikeItem | tuple[_SliceLikeItem, ...]


JaxBooleanArray = JaxArray
JaxIntegralArray = JaxArray
JaxRealArray = JaxArray
JaxComplexArray = JaxArray
KeyArray = JaxArray


NumpyArray = npt.NDArray[Any]
NumpyBooleanArray = npt.NDArray[np.bool_]
NumpyIntegralArray = npt.NDArray[np.integer[Any]]
NumpyRealArray = npt.NDArray[np.floating[Any]]
NumpyComplexArray = npt.NDArray[np.floating[Any] | np.complexfloating[Any, Any]]
Array = NumpyArray | JaxArray
BooleanArray = NumpyBooleanArray | JaxArray
IntegralArray = NumpyIntegralArray | JaxArray
RealArray = NumpyRealArray | JaxArray
ComplexArray = NumpyComplexArray | JaxArray


Integral = int
Real = float | int
Complex = complex | float | int
NumpyBooleanNumeric = NumpyBooleanArray | bool
NumpyIntegralNumeric = NumpyIntegralArray | Integral
NumpyRealNumeric = NumpyRealArray | Real
NumpyComplexNumeric = NumpyComplexArray | Complex
BooleanNumeric = BooleanArray | bool
IntegralNumeric = IntegralArray | Integral
RealNumeric = RealArray | Real
ComplexNumeric = ComplexArray | Complex


PyTree = Any
TapFunctionTransforms = Sequence[tuple[str, dict[str, Any]]]
