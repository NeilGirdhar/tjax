from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Hashable, List, Tuple, Union

import numpy as np
from jax import numpy as jnp
from numpy.typing import _Shape as Shape
from numpy.typing import _ShapeLike as ShapeLike

__all__ = ['Shape',
           'ShapeLike',
           'SliceLike',
           'Tensor',
           'RealTensor',
           'ComplexTensor',
           'PyTree']


SliceLike = Tuple[Union[int, None, slice], ...]
Tensor = Union[np.ndarray, jnp.ndarray]
RealTensor = Tensor
ComplexTensor = Tensor
PyTree = Union[Tensor,
               'PyTreeLike',
               Tuple['PyTree', ...],
               List['PyTree'],
               Dict[Hashable, 'PyTree'],
               None]


if TYPE_CHECKING:
    from .pytree_like import PyTreeLike
