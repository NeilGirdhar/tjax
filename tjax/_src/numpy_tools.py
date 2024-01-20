import numpy as np

from .annotations import NumpyComplexArray, NumpyRealArray

__all__ = ['create_diagonal_array', 'np_abs_square']


def np_abs_square(x: NumpyComplexArray) -> NumpyRealArray:
    return np.square(x.real) + np.square(x.imag)  # pyright: ignore


def create_diagonal_array(m: NumpyRealArray) -> NumpyRealArray:
    """A vectorized version of diagonal.

    Args:
        m: Has shape (*k, n)
    Returns: Array with shape (*k, n, n) and the elements of m on the diagonals.
    """
    indices = (..., *np.diag_indices(m.shape[-1]))
    retval = np.zeros((*m.shape, m.shape[-1]), dtype=m.dtype)
    retval[indices] = m
    return retval
