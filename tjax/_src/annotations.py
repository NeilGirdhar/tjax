from __future__ import annotations

from collections.abc import Sequence
from types import ModuleType
from typing import Any

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc
from jax import Array as JaxArray

type Shape = tuple[int, ...]
type ShapeLike = int | Sequence[int]
type _SliceLikeItem = int | slice | None
type SliceLike = _SliceLikeItem | tuple[_SliceLikeItem, ...]


type JaxBooleanArray = JaxArray
type JaxIntegralArray = JaxArray
type JaxRealArray = JaxArray
type JaxComplexArray = JaxArray
type KeyArray = JaxArray


type NumpyArray = onp.ArrayND
type NumpyBooleanArray = onp.ArrayND[np.bool_]
type NumpyIntegralArray = onp.ArrayND[npc.integer]
type NumpyRealArray = onp.ArrayND[npc.floating]
type NumpyComplexArray = onp.ArrayND[npc.number]
type BooleanArray = NumpyBooleanArray | JaxArray
type IntegralArray = NumpyIntegralArray | JaxArray
type RealArray = NumpyRealArray | JaxArray
type ComplexArray = NumpyComplexArray | JaxArray


type NumpyBooleanNumeric = NumpyBooleanArray | bool
type NumpyIntegralNumeric = NumpyIntegralArray | int
type NumpyRealNumeric = NumpyRealArray | float
type NumpyComplexNumeric = NumpyComplexArray | complex
type BooleanNumeric = BooleanArray | bool
type IntegralNumeric = IntegralArray | int
type RealNumeric = RealArray | float
type ComplexNumeric = ComplexArray | complex


# Eventually, these will come from array_api_extra.
type Array = NumpyArray | JaxArray
type Device = Any
type DType = Any
type Namespace = ModuleType


type PyTree = Any
