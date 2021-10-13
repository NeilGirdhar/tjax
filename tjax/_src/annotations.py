from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

__all__ = ['Shape', 'ShapeLike', 'SliceLike', 'Array', 'BooleanArray', 'IntegralArray', 'RealArray',
           'ComplexArray', 'Integral', 'Real', 'Complex', 'BooleanNumeric', 'IntegralNumeric',
           'RealNumeric', 'ComplexNumeric', 'PyTree', 'TapFunctionTransforms']

Shape = Tuple[int, ...]
ShapeLike = Union[int, Sequence[int]]
_SliceLikeItem = Union[int, None, slice]
SliceLike = Union[_SliceLikeItem, Tuple[_SliceLikeItem, ...]]


Array = npt.NDArray[Any]
BooleanArray = npt.NDArray[np.bool_]
if TYPE_CHECKING:
    IntegralArray = npt.NDArray[np.integer[Any]]
    RealArray = npt.NDArray[np.floating[Any]]
    ComplexArray = npt.NDArray[Union[np.floating[Any], np.complexfloating[Any, Any]]]
else:
    IntegralArray = npt.NDArray[np.integer]
    RealArray = npt.NDArray[np.floating]
    ComplexArray = npt.NDArray[Union[np.floating, np.complexfloating]]


Integral = int  # Use this until numbers.Integral works with MyPy.
Real = Union[float, int]  # Use this until numbers.Real works with MyPy.
Complex = Union[complex, float, int]  # Use this until numbers.Complex works with MyPy.
BooleanNumeric = Union[BooleanArray, bool]
IntegralNumeric = Union[IntegralArray, Integral]
RealNumeric = Union[RealArray, Real]
ComplexNumeric = Union[ComplexArray, Complex]


PyTree = Any
TapFunctionTransforms = Sequence[Tuple[str, Dict[str, Any]]]
