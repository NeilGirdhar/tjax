from __future__ import annotations

from typing import Any, Mapping, Type

import jax.numpy as jnp
import numpy as np
from jax.dtypes import canonicalize_dtype

__all__ = ['int_dtype',
           'real_dtype',
           'complex_dtype',
           'default_rtol',
           'default_atol',
           'default_tols']


def int_dtype() -> Type[np.signedinteger[Any]]:
    """
    Returns: The type of the dtype used by JAX for int values.  Typically, either `numpy.int32.type`
        or `numpy.int64.type`.
    """
    return canonicalize_dtype(jnp.int_).type


def real_dtype() -> Type[np.floating[Any]]:
    """
    Returns: The type of the dtype used by JAX for int values.  Typically, either
        `numpy.float32.type` or `numpy.float64.type`.
    """
    return canonicalize_dtype(jnp.float_).type


def complex_dtype() -> Type[np.complexfloating[Any, Any]]:
    """
    Returns: The type of the dtype used by JAX for int values.  Typically, either
        `numpy.complex64.type` or `numpy.complex128.type`.
    """
    return canonicalize_dtype(jnp.complex_).type


def default_rtol() -> float:
    return {jnp.float32: 1e-4, jnp.float64: 1e-5}[real_dtype()]
def default_atol() -> float:
    return {jnp.float32: 1e-6, jnp.float64: 1e-8}[real_dtype()]
def default_tols() -> Mapping[str, float]:
    return dict(rtol=default_rtol(), atol=default_atol())
