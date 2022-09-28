from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
from jax import Array as JaxArray

__all__ = ['Shape', 'ShapeLike', 'SliceLike', 'Array', 'BooleanArray', 'IntegralArray', 'RealArray',
           'ComplexArray', 'Integral', 'Real', 'Complex', 'BooleanNumeric', 'IntegralNumeric',
           'RealNumeric', 'ComplexNumeric', 'PyTree', 'TapFunctionTransforms', 'NumpyArray',
           'NumpyBooleanArray', 'NumpyIntegralArray', 'NumpyRealArray', 'NumpyComplexArray',
           'NumpyBooleanNumeric', 'NumpyIntegralNumeric', 'NumpyRealNumeric', 'NumpyComplexNumeric']

Shape = Tuple[int, ...]
ShapeLike = Union[int, Sequence[int]]
_SliceLikeItem = Union[int, None, slice]
SliceLike = Union[_SliceLikeItem, Tuple[_SliceLikeItem, ...]]


NumpyArray = npt.NDArray[Any]
NumpyBooleanArray = npt.NDArray[np.bool_]
NumpyIntegralArray = npt.NDArray[np.integer[Any]]
NumpyRealArray = npt.NDArray[np.floating[Any]]
NumpyComplexArray = npt.NDArray[Union[np.floating[Any], np.complexfloating[Any, Any]]]
Array = Union[NumpyArray, JaxArray]
BooleanArray = Union[NumpyBooleanArray, JaxArray]
IntegralArray = Union[NumpyIntegralArray, JaxArray]
RealArray = Union[NumpyRealArray, JaxArray]
ComplexArray = Union[NumpyComplexArray, JaxArray]


Integral = int  # Use this until numbers.Integral works with MyPy.
Real = Union[float, int]  # Use this until numbers.Real works with MyPy.
Complex = Union[complex, float, int]  # Use this until numbers.Complex works with MyPy.
NumpyBooleanNumeric = Union[NumpyBooleanArray, bool]
NumpyIntegralNumeric = Union[NumpyIntegralArray, Integral]
NumpyRealNumeric = Union[NumpyRealArray, Real]
NumpyComplexNumeric = Union[NumpyComplexArray, Complex]
BooleanNumeric = Union[BooleanArray, bool]
IntegralNumeric = Union[IntegralArray, Integral]
RealNumeric = Union[RealArray, Real]
ComplexNumeric = Union[ComplexArray, Complex]


PyTree = Any
TapFunctionTransforms = Sequence[Tuple[str, Dict[str, Any]]]
