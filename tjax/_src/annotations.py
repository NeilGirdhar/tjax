from __future__ import annotations

from collections.abc import Sequence
from types import ModuleType
from typing import Any, TypeAlias

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc
from jax import Array as JaxArray

Shape: TypeAlias = tuple[int, ...]
ShapeLike: TypeAlias = int | Sequence[int]
_SliceLikeItem: TypeAlias = int | slice | None
SliceLike: TypeAlias = _SliceLikeItem | tuple[_SliceLikeItem, ...]


JaxBooleanArray: TypeAlias = JaxArray
JaxIntegralArray: TypeAlias = JaxArray
JaxRealArray: TypeAlias = JaxArray
JaxComplexArray: TypeAlias = JaxArray
KeyArray: TypeAlias = JaxArray


NumpyArray: TypeAlias = onp.ArrayND
NumpyBooleanArray: TypeAlias = onp.ArrayND[np.bool_]
NumpyIntegralArray: TypeAlias = onp.ArrayND[npc.integer]
NumpyRealArray: TypeAlias = onp.ArrayND[npc.floating]
NumpyComplexArray: TypeAlias = onp.ArrayND[npc.number]
BooleanArray: TypeAlias = NumpyBooleanArray | JaxArray
IntegralArray: TypeAlias = NumpyIntegralArray | JaxArray
RealArray: TypeAlias = NumpyRealArray | JaxArray
ComplexArray: TypeAlias = NumpyComplexArray | JaxArray


NumpyBooleanNumeric: TypeAlias = NumpyBooleanArray | bool
NumpyIntegralNumeric: TypeAlias = NumpyIntegralArray | int
NumpyRealNumeric: TypeAlias = NumpyRealArray | float
NumpyComplexNumeric: TypeAlias = NumpyComplexArray | complex
BooleanNumeric: TypeAlias = BooleanArray | bool
IntegralNumeric: TypeAlias = IntegralArray | int
RealNumeric: TypeAlias = RealArray | float
ComplexNumeric: TypeAlias = ComplexArray | complex


# Eventually, these will come from array_api_extra.
Array: TypeAlias = NumpyArray | JaxArray
Device: TypeAlias = Any
DType: TypeAlias = Any
Namespace: TypeAlias = ModuleType


PyTree: TypeAlias = Any
