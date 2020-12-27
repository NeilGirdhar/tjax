from functools import partial, singledispatch
from numbers import Complex, Number, Real
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
from jax import numpy as jnp
from jax.interpreters.xla import DeviceArray
from jax.tree_util import tree_multimap, tree_reduce

from .annotations import PyTree
from .dtypes import default_atol, default_rtol

__all__ = ['assert_jax_allclose', 'jax_allclose', 'get_test_string', 'get_relative_test_string']


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
    from tjax import assert_jax_allclose, dataclass, Array

    @dataclass
    class A:
        x: Array
        y: Array

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
    except AssertionError:
        print("JAX trees don't match.\nActual:")
        print(actual)
        print("Desired:")
        print(desired)
        print("Test string:")
        if original_name is not None and original_value is not None:
            print(get_relative_test_string(actual, original_name, original_value, rtol, atol))
        else:
            print(get_test_string(actual, rtol, atol))
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


# get test string ----------------------------------------------------------------------------------
# Redefinition typing errors in this file are due to https://github.com/python/mypy/issues/2904.
@singledispatch
def get_test_string(actual: Any, rtol: float, atol: float) -> str:
    """
    Args:
        actual: The actual value that was produced, and that should be the desired value.
        rtol: The relative tolerance of the comparisons in the assertion.
        atol: The absolute tolerance of the comparisons in the assertion.
    Returns:
        A string of Python code that produces the desired value.
    """
    return str(actual)


@get_test_string.register(np.ndarray)
@get_test_string.register(DeviceArray)
def _(actual: Union[np.ndarray, DeviceArray], rtol: float, atol: float) -> str:
    def fts(x: Complex) -> str:
        return _float_to_string(x, rtol, atol)
    with np.printoptions(formatter={'float_kind': fts,
                                    'complex_kind': fts}):
        return "np." + repr(np.asarray(actual)).replace(' ]', ']').replace(' ,', ',').replace(
            '  ', ' ')


@get_test_string.register  # type: ignore
def _(actual: Complex, rtol: float, atol: float) -> str:
    return _float_to_string(complex(actual), rtol, atol)


@get_test_string.register  # type: ignore
def _(actual: Number, rtol: float, atol: float) -> str:
    return str(actual)


@get_test_string.register(list)  # type: ignore
@get_test_string.register(tuple)
def _(actual: Union[List[Any], Tuple[Any]], rtol: float, atol: float) -> str:
    is_list = isinstance(actual, list)
    return (("[" if is_list else "(")
            + ", ".join(get_test_string(sub_actual, rtol, atol)
                        for i, sub_actual in enumerate(actual))
            + ("]" if is_list else ")"))


@get_test_string.register(dict)  # type: ignore
def _(actual: Dict[Any, Any], rtol: float, atol: float) -> str:
    return '{' + ",\n".join(repr(key) + ': ' + get_test_string(sub_actual, rtol, atol)
                            for key, sub_actual in actual.items()) + '}'


# get relative test string -------------------------------------------------------------------------
@singledispatch
def get_relative_test_string(actual: Any,
                             original_name: str,
                             original: Any,
                             rtol: float,
                             atol: float) -> str:
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
    return str(actual)


@get_relative_test_string.register(np.ndarray)  # type: ignore
@get_relative_test_string.register(DeviceArray)
def _(actual: Union[np.ndarray, DeviceArray], original_name: str, original: Any, rtol: float,
      atol: float) -> str:
    def fts(x: Complex) -> str:
        return _float_to_string(x, rtol, atol)
    with np.printoptions(formatter={'float_kind': fts,
                                    'complex_kind': fts}):
        return "np." + repr(np.asarray(actual)).replace(' ]', ']').replace(' ,', ',').replace(
            '  ', ' ')


@get_relative_test_string.register  # type: ignore
def _(actual: Complex, original_name: str, original: Any, rtol: float, atol: float) -> str:
    return _float_to_string(actual, rtol, atol)


@get_relative_test_string.register  # type: ignore
def _(actual: Number, original_name: str, original: Any, rtol: float, atol: float) -> str:
    return str(actual)


@get_relative_test_string.register(list)  # type: ignore
@get_relative_test_string.register(tuple)
def _(actual: Union[List[Any], Tuple[Any]], original_name: str, original: Any, rtol: float,
      atol: float) -> str:
    is_list = isinstance(actual, list)
    return (("[" if is_list else "(")
            + ", ".join(get_relative_test_string(f"{original_name}[{i}]",
                                                 sub_actual, sub_original,
                                                 rtol, atol)
                        for i, (sub_actual, sub_original) in enumerate(zip(actual, original)))
            + ("]" if is_list else ")"))


@get_relative_test_string.register(dict)  # type: ignore
def _(actual: Dict[Any, Any], original_name: str, original: Any, rtol: float, atol: float) -> str:
    if not isinstance(original, dict):
        raise TypeError

    def relative_string(key, sub_actual):
        return get_relative_test_string(
            f"{original_name}[{key}]", sub_actual, original[key], rtol, atol)

    return '{' + ",\n".join(repr(key) + ': ' + relative_string(key, sub_actual)
                            for key, sub_actual in actual.items()) + '}'


# Private functions --------------------------------------------------------------------------------
def _float_to_string_with_precision(x: Union[float, complex], precision: int) -> str:
    with np.printoptions(precision=precision, floatmode='maxprec'):
        return repr(np.array(x))[6:-1]


def _float_to_string(x: complex, rtol: float, atol: float) -> str:
    y: Union[float, complex]
    if isinstance(x, Real):
        y = float(x)
    elif isinstance(x, Complex):
        y = complex(x)
    else:
        raise TypeError
    for i in range(20):
        retval = _float_to_string_with_precision(y, i)
        if np.allclose(complex(retval), x, rtol=rtol, atol=atol):
            break
    return retval
