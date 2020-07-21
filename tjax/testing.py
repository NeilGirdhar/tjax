from functools import partial
from numbers import Number
from typing import Any, Optional, cast

import numpy as np
from jax import numpy as jnp
from jax.interpreters.xla import DeviceArray
from jax.tree_util import tree_multimap, tree_reduce

from .annotations import PyTree
from .dtypes import default_atol, default_rtol

__all__ = ['assert_jax_allclose', 'jax_allclose']


def assert_jax_allclose(actual: PyTree,
                        desired: PyTree,
                        original_name: Optional[str] = None,
                        original_value: Optional[PyTree] = None,
                        *,
                        rtol: Optional[float] = None,
                        atol: Optional[float] = None) -> None:
    """
    Asserts that every tensor in an actual pytree matches the corresponding tensor in a desired
    pytree.  If the assertion fails, a passing test string is printed::

    ```python
    from tjax import assert_jax_allclose, dataclass, Tensor

    @dataclass
    class A:
        x: Tensor
        y: Tensor

    @dataclass
    class B:
        z: A

    original = B(A(1.2, 3.4))
    desired = B(A(3.0, 4.0))
    actual = B(A(1.2, 5.2))

    assert_jax_allclose(actual, desired, 'original', original)
    ```
    This prints::
    ```
    JAX trees don't match.  Actual:
    B
        z=A
            x=3.0
            y=4.0

    Desired:
    B
        z=A
            x=1.2
            y=5.2

    Test string:
    original.replace(z=original.z.replace(x=3.0, y=4.0))
    ```
    The test string can then be pasted.

    Args:
        actual: The actual value.
        desired: The desired value.
        original_name: The variable name that contains the original value.
        original_value: The original value.  This is usually a pytree like a dataclass that has the
            same type as actual and desired, but contains different values.
        rtol: The relative tolerance of the comparisons in the assertion.
        atol: The absolute tolerance of the comparisons in the assertion.
    """
    if rtol is None:
        rtol = default_rtol
    if atol is None:
        atol = default_atol

    try:
        tree_multimap(partial(np.testing.assert_allclose, rtol=rtol, atol=atol), actual, desired)
    except Exception:
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
    """
    Args:
        actual: The actual value.
        desired: The desired value.
        rtol: The relative tolerance of the comparisons in the comparison.
        atol: The absolute tolerance of the comparisons in the comparison.
    """
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
    """
    Args:
        original_name: The name of the variable containing an original value.
        actual: The actual value that was produced, and that should be the desired value.
        original: The original value.
        rtol: The relative tolerance of the comparisons in the assertion.
        atol: The absolute tolerance of the comparisons in the assertion.
    Returns:
        A string of Python code that produces the desired value from an "original value" (could be
        zeroed-out, for example).
    """
    if isinstance(actual, (np.ndarray, DeviceArray)):
        with np.printoptions(precision=int(np.ceil(-np.log10(default_rtol))),
                             floatmode='maxprec'):
            return "np." + repr(np.asarray(actual)).replace(' ]', ']').replace(' ,', ',').replace(
                '  ', ' ')
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
    if isinstance(actual, (list, tuple)):
        is_list = isinstance(actual, list)
        return (("[" if is_list else "(")
                + ", ".join(get_test_string(f"{original_name}[{i}]", sub_actual, sub_original, rtol,
                                            atol)
                            for i, (sub_actual, sub_original) in enumerate(zip(actual, original)))
                + ("]" if is_list else ")"))
    if isinstance(actual, dict):
        if not isinstance(original, dict):
            raise TypeError
        return str({key: get_test_string(f"{original_name}[{key}]", sub_actual, original[key],
                                         rtol, atol)
                    for key, sub_actual in actual.items()})
    return str(actual)
