from __future__ import annotations

from typing import Any, TypedDict

import jax.numpy as jnp

__all__ = ['default_rtol',
           'default_atol',
           'default_tols',
           'int_dtype',
           'float_dtype',
           'complex_dtype']


int_dtype = jnp.asarray(0).dtype
float_dtype = jnp.empty(1).dtype
complex_dtype = jnp.asarray(1j).dtype


class Tols(TypedDict):
    rtol: float
    atol: float


rtol_dict = {jnp.bfloat16: 1e-4,
             jnp.float16: 1e-3, jnp.float32: 1e-4, jnp.float64: 1e-5,
             jnp.complex64: 1e-4, jnp.complex128: 1e-5}
atol_dict = {jnp.bfloat16: 1e-2,
             jnp.float16: 1e-3, jnp.float32: 1e-6, jnp.float64: 1e-8,
             jnp.complex64: 1e-6, jnp.complex128: 1e-8}


def default_rtol(dtype: type[jnp.number[Any]]) -> float:
    if not jnp.issubdtype(dtype, jnp.inexact):
        return 0.0
    return rtol_dict[dtype]  # type: ignore[index] # pyright: ignore


def default_atol(dtype: type[jnp.number[Any]]) -> float:
    if not jnp.issubdtype(dtype, jnp.inexact):
        return 1.0
    return atol_dict[dtype]  # type: ignore[index] # pyright: ignore


def default_tols(dtype: type[jnp.number[Any]],
                 *,
                 rtol: float | None = None,
                 atol: float | None = None) -> Tols:
    return Tols(rtol=default_rtol(dtype) if rtol is None else rtol,
                atol=default_atol(dtype) if atol is None else atol)
