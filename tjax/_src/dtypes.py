from __future__ import annotations

from typing import Any, TypeAlias, TypedDict

import jax.numpy as jnp

__all__ = ['default_rtol',
           'default_atol',
           'default_tols']


class Tols(TypedDict):
    rtol: float
    atol: float


# Work around jax.numpy._ScalarMeta
JNumber: TypeAlias = type[jnp.number[Any]]
rtol_dict: dict[JNumber, float] = {jnp.bfloat16.dtype.type: 1e-4, jnp.float16.dtype.type: 1e-3,
                                   jnp.float32.dtype.type: 1e-4, jnp.float64.dtype.type: 1e-5,
                                   jnp.complex64.dtype.type: 1e-4, jnp.complex128.dtype.type: 1e-5}
atol_dict: dict[JNumber, float] = {jnp.bfloat16.dtype.type: 1e-2, jnp.float16.dtype.type: 1e-3,
                                   jnp.float32.dtype.type: 1e-6, jnp.float64.dtype.type: 1e-8,
                                   jnp.complex64.dtype.type: 1e-6, jnp.complex128.dtype.type: 1e-8}


def default_rtol(dtype: JNumber) -> float:
    if not jnp.issubdtype(dtype, jnp.inexact):
        return 0.0
    return rtol_dict[dtype]


def default_atol(dtype: JNumber) -> float:
    if not jnp.issubdtype(dtype, jnp.inexact):
        return 1.0
    return atol_dict[dtype]


def default_tols(dtype: type[jnp.number[Any]],
                 *,
                 rtol: float | None = None,
                 atol: float | None = None) -> Tols:
    return Tols(rtol=default_rtol(dtype) if rtol is None else rtol,
                atol=default_atol(dtype) if atol is None else atol)
