from __future__ import annotations

from typing import TypeVar

import jax
import jax.numpy as jnp
from jax.dtypes import canonicalize_dtype

_T = TypeVar('_T', bound=tuple[jax.Array | None, ...])


def result_type(*args: jax.Array | None,
                dtype: jax.typing.DTypeLike | None = None,
                ensure_inexact: bool = True
                ) -> jax.typing.DTypeLike:
    """Find a common data type for arrays.

    Args:
        *args: Arrays to consider (or None).
        dtype: Overrides the inferred data type.
        ensure_inexact: Ensures that the result type is inexact, or raises.

    Returns:
        The common data type of *args.
    """
    if dtype is None:
        filtered_args = [x for x in args if x is not None]
        dtype = jnp.result_type(*filtered_args)
        if ensure_inexact and not jnp.issubdtype(dtype, jnp.inexact):
            dtype = jnp.promote_types(canonicalize_dtype(float), dtype)
    if ensure_inexact and not jnp.issubdtype(dtype, jnp.inexact):
        msg = f'Data type must be inexact: {dtype}'
        raise ValueError(msg)
    return dtype


def cast_to_result_type(arrays: _T,
                        /,
                        *,
                        dtype: jax.typing.DTypeLike | None = None,
                        ensure_inexact: bool = True
                        ) -> _T:
    """Casts the input arrays to a common data dtype.

    Args:
        arrays: Arrays to be converted (or None).
        dtype: Overrides the inferred data type.
        ensure_inexact: Ensures that the result type is inexact, or raises.

    Returns:
        The arrays cast to the inferred type.
    """
    dtype = result_type(*arrays, dtype=dtype, ensure_inexact=ensure_inexact)
    return tuple(jnp.asarray(x, dtype)  # type: ignore # pyright: ignore
                 if x is not None else None for x in arrays)
