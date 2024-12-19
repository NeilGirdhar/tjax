from math import prod

import numpy as np

from tjax import create_diagonal_array


def test_create_diagonal_array() -> None:
    s = (2, 3, 4)
    x = np.arange(prod(s), dtype=np.float64)
    y = np.reshape(x, s)
    z = create_diagonal_array(y)
    assert z.shape == (*s, s[-1])  # type: ignore[comparison-overlap]
    for index, value in np.ndenumerate(z):
        if index[-2] == index[-1]:
            y_index = index[:-1]
            assert y[y_index] == value
        else:
            assert value == 0
