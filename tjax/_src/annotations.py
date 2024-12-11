from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from jax import Array as JaxArray

Shape = tuple[int, ...]
ShapeLike = int | Sequence[int]
_SliceLikeItem = int | slice | None
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
NumpyComplexArray = npt.NDArray[np.floating[Any]] | npt.NDArray[np.complexfloating[Any, Any]]
Array = NumpyArray | JaxArray
BooleanArray = NumpyBooleanArray | JaxArray
IntegralArray = NumpyIntegralArray | JaxArray
RealArray = NumpyRealArray | JaxArray
ComplexArray = NumpyComplexArray | JaxArray


NumpyBooleanNumeric = NumpyBooleanArray | bool
NumpyIntegralNumeric = NumpyIntegralArray | int
NumpyRealNumeric = NumpyRealArray | float
NumpyComplexNumeric = NumpyComplexArray | complex
BooleanNumeric = BooleanArray | bool
IntegralNumeric = IntegralArray | int
RealNumeric = RealArray | float
ComplexNumeric = ComplexArray | complex


PyTree = Any
