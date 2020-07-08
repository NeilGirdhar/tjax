from functools import partial
from numbers import Number
from typing import Any, Optional, cast

import numpy as np
from jax import numpy as jnp
from jax.interpreters.xla import DeviceArray
from jax.tree_util import tree_multimap, tree_reduce

from .annotations import PyTree
from .dtypes import default_atol, default_rtol

__all__ = ['assert_jax_allclose', 'jax_allclose', 'get_test_string']


def assert_jax_allclose(actual: PyTree,
                        desired: PyTree,
                        original_name: Optional[str] = None,
                        original_value: Optional[PyTree] = None,
                        *,
                        rtol: Optional[float] = None,
                        atol: Optional[float] = None) -> None:
    if rtol is None:
        rtol = default_rtol
    if atol is None:
        atol = default_atol

    try:
        tree_multimap(partial(np.testing.assert_allclose, rtol=rtol, atol=atol), actual, desired)
    except:
        print("JAX trees don't match.  Actual:")
        print(actual)
        print("Desired:")
        print(desired)
        if original_name is not None and original_value is not None:
            print("Test string:")
            print(get_test_string(original_name, actual, original_value, rtol, atol))
        raise


def jax_allclose(actual: PyTree,
                 desired: PyTree,
                 rtol: Optional[float] = None,
                 atol: Optional[float] = None) -> bool:
    if rtol is None:
        rtol = default_rtol
    if atol is None:
        atol = default_atol

    return cast(
        bool,
        tree_reduce(jnp.logical_and,
                    tree_multimap(partial(np.allclose, rtol=rtol, atol=atol), actual, desired),
                    True))


def get_test_string(original_name: str,
                    actual: Any,
                    original: Any,
                    rtol: Optional[float] = None,
                    atol: Optional[float] = None) -> str:
    if isinstance(actual, (np.ndarray, DeviceArray)):
        return "np." + repr(np.asarray(actual))
    if isinstance(actual, Number):
        return str(actual)
    if hasattr(actual, 'display'):
        retval = f"{original_name}.replace("
        retval += ",\n".join(
            f"{fn}=" + get_test_string(f"{original_name}.{fn}",
                                       getattr(actual, fn),
                                       getattr(original, fn),
                                       rtol,
                                       atol)
            for fn in actual.tree_fields
            if not jax_allclose(getattr(actual, fn), getattr(original, fn), rtol=rtol, atol=atol))
        retval += ")"
        return retval
    if isinstance(actual, dict):
        return str({key: get_test_string(f"{original_name}[{key}]", sub_value, original[key],
                                         rtol, atol)
                    for key, sub_value in actual.items()})
    return str(actual)
