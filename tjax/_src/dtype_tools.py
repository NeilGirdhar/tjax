from __future__ import annotations

from typing import TypeVar

import jax
import jax.numpy as jnp


_T = TypeVar('_T', bound=tuple[jax.Array | None, ...])


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
    dtype = jnp.result_type(*arrays, dtype, *([float] if ensure_inexact else []))
    return tuple(jnp.asarray(x, dtype)  # type: ignore # pyright: ignore
                 if x is not None else None for x in arrays)
