from __future__ import annotations

from typing import Any, Mapping, Optional, Type

import jax.numpy as jnp
import numpy as np

__all__ = ['default_rtol',
           'default_atol',
           'default_tols']


def default_rtol(dtype: Type[np.floating[Any]]) -> float:
    return {jnp.float32: 1e-4, jnp.float64: 1e-5}[dtype]


def default_atol(dtype: Type[np.floating[Any]]) -> float:
    return {jnp.float32: 1e-6, jnp.float64: 1e-8}[dtype]


def default_tols(dtype: Type[np.floating[Any]],
                 *,
                 rtol: Optional[float] = None,
                 atol: Optional[float] = None) -> Mapping[str, float]:
    return dict(rtol=default_rtol(dtype) if rtol is None else rtol,
                atol=default_atol(dtype) if atol is None else atol)
