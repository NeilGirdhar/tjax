from __future__ import annotations

from typing import Any, Optional, Tuple

from jax.tree_util import tree_leaves

__all__ = ['BatchDimensionIterator']


class BatchDimensionIterator:
    """
    Batch dimensions is created by using jax.vmap on a function.  V-mapped dimensions can have
    non-unity batch dimensions.  This class keeps track of these dimensions and provides
    iterator-like behavior for displaying the batch dimensions along corresponding elements.
    """
    def __init__(self, batch_dims: Optional[Tuple[Optional[int], ...]] = None):
        """
        Args:
            batch_dims: A tuple of integers corresponding to the leaves of the PyTree.  This class
                keeps track of these dimensions
        """
        super().__init__()
        self.batch_dims = batch_dims
        self.i = 0

    def advance(self, value: Any) -> Optional[Tuple[Optional[int], ...]]:
        """
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
        """
        Asserts that all of the batch dimensions are accounted for.
        """
        if self.batch_dims is not None:
            assert self.i == len(self.batch_dims)
