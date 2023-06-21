from __future__ import annotations

from typing import Any, Optional, Tuple

from jax.tree_util import tree_leaves
from typing_extensions import TypeAlias, override

__all__ = ['BatchDimensionIterator', 'BatchDimensions']


_SimpleBatchDimensions: TypeAlias = Optional[Tuple[Optional[int], ...]]
BatchDimensions: TypeAlias = Optional[Tuple[Optional[Tuple[int, ...]], ...]]


def combine_batch_dimensions(x: BatchDimensions, y: _SimpleBatchDimensions) -> BatchDimensions:
    if y is None:
        return x
    # Convert y to BatchDimensions.
    y_bd = tuple(None if yi is None else (yi,)
                 for yi in y)
    if x is None:
        return y_bd
    # Merge x and y.
    assert len(x) == len(y)
    return tuple(yi if xi is None
                 else xi if yi is None
                 else xi + yi
                 for xi, yi in zip(x, y_bd, strict=True))


class BatchDimensionIterator:
    """An iterator for batch dimensions.

    Batch dimensions is created by using jax.vmap on a function.  V-mapped dimensions can have
    non-unity batch dimensions.  This class keeps track of these dimensions and provides
    iterator-like behavior for displaying the batch dimensions along corresponding elements.

    Args:
        batch_dims: A tuple of integers corresponding to the leaves of the PyTree.  This class
        keeps track of these dimensions.
    """
    @override
    def __init__(self, batch_dims: BatchDimensions | None = None):
        super().__init__()
        self.batch_dims = batch_dims
        self.i = 0

    def advance(self, value: Any) -> BatchDimensions:
        """Advance the iterator.

        Args:
            value: The next sub-element of the PyTree.  It need not be a leaf.
        Returns: The batch dimensions of the leaves of value.
        """
        if self.batch_dims is None:
            return None
        n = len(tree_leaves(value))
        old_i = self.i
        self.i += n
        return self.batch_dims[old_i: self.i]

    def check_done(self) -> None:
        """Asserts that all of the batch dimensions are accounted for."""
        if self.batch_dims is not None:
            assert self.i == len(self.batch_dims)
