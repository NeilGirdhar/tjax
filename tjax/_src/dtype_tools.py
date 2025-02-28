from __future__ import annotations

from typing import TypeVar

from array_api_compat import array_namespace

from .annotations import Array, DType, Namespace

_T = TypeVar('_T', bound=tuple[Array | None, ...])


def cast_to_result_type(arrays: _T,
                        *arrays_and_dtypes: type[complex | bool] | Array | DType,
                        xp: Namespace | None = None
                        ) -> _T:
    """Casts the input arrays to a common data dtype.

    Args:
        arrays: Arrays to be converted (or None).
        arrays_and_dtypes: Extra arrays and types to be passed to result_type.
        xp: The optional namespace.

    Returns:
        The arrays cast to the inferred type.
    """
    if xp is None:
        xp = array_namespace(*arrays)
    dtype = xp.result_type(*arrays, *arrays_and_dtypes)
    return tuple(xp.asarray(x, dtype)  # type: ignore # pyright: ignore
                 if x is not None else None for x in arrays)
