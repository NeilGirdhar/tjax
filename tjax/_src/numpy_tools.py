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
    pre = m.shape[:-1]
    n = m.shape[-1]
    s = (*m.shape, n)
    retval = np.zeros((*pre, n ** 2), dtype=m.dtype)
    for index in np.ndindex(*pre):
        retval[(*index, slice(None, None, n + 1))] = m[
                (*index, slice(None))]  # type:ignore[arg-type]
    return np.reshape(retval, s)
