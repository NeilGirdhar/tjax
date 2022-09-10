from __future__ import annotations

from typing import Any, Optional, Type, TypedDict

import jax.numpy as jnp

__all__ = ['default_rtol',
           'default_atol',
           'default_tols']


class Tols(TypedDict):
    rtol: float
    atol: float


def default_rtol(dtype: Type[jnp.number[Any]]) -> float:
    if not jnp.issubdtype(dtype, jnp.inexact):  # type: ignore[no-untyped-call]
        return 0.0
    return {jnp.bfloat16: 1e-4,
            jnp.float16: 1e-3, jnp.float32: 1e-4, jnp.float64: 1e-5,
            jnp.complex64: 1e-4, jnp.complex128: 1e-5}[dtype]  # pyright: ignore


def default_atol(dtype: Type[jnp.number[Any]]) -> float:
    if not jnp.issubdtype(dtype, jnp.inexact):  # type: ignore[no-untyped-call]
        return 1.0
    return {jnp.bfloat16: 1e-2,
            jnp.float16: 1e-3, jnp.float32: 1e-6, jnp.float64: 1e-8,
            jnp.complex64: 1e-6, jnp.complex128: 1e-8}[dtype]  # pyright: ignore


def default_tols(dtype: Type[jnp.number[Any]],
                 *,
                 rtol: Optional[float] = None,
                 atol: Optional[float] = None) -> Tols:
    return Tols(rtol=default_rtol(dtype) if rtol is None else rtol,
                atol=default_atol(dtype) if atol is None else atol)
