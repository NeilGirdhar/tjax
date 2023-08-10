from __future__ import annotations

from functools import partial, singledispatch
from math import prod
from numbers import Complex, Integral, Real
from operator import and_
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_flatten, tree_map, tree_reduce

from .annotations import Array, PyTree
from .dtypes import default_tols

__all__ = ['assert_tree_allclose', 'tree_allclose', 'get_test_string', 'get_relative_test_string']


def assert_tree_allclose(actual: PyTree,
                         desired: PyTree,
                         original_name: str | None = None,
                         original_value: PyTree | None = None,
                         *,
                         rtol: float | None = None,
                         atol: float | None = None) -> None:
    """Assert that an actual pytree matches a desired pytree.

    If the assertion fails, a passing test string is printed::

    ```python
    from tjax import assert_tree_allclose, RealNumeric
    from tjax.dataclasses import dataclass

    @dataclass
    class A:
        x: RealNumeric
        y: RealNumeric

    @dataclass
    class B:
        z: A

    original = B(A(1.2, 3.4))
    desired = B(A(3.0, 4.0))
    actual = B(A(1.2, 5.2))

    assert_tree_allclose(actual, desired)
    ```
    This prints::
    ```
    Tree leaves don't match at position 0 with rtol=0.0001 and atol=1e-06.
    Mismatched elements: 1 / 1 (100%)
    Maximum absolute difference: 1.8
    Maximum relative difference: 0.6

    Actual: B(z=A(x=1.2, y=5.2))
    Desired: B(z=A(x=3.0, y=4.0))
    Test string:
    desired = B(A(1.2, 5.2))
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
    flattened_actual, structure_actual = tree_flatten(actual)
    flattened_desired, structure_desired = tree_flatten(desired)
    if structure_actual != structure_desired:
        msg = f"\nTree structure mismatch.\nActual: {actual}\nDesired: {desired}\n"
        raise AssertionError(msg)

    for i, (actual_, desired_) in enumerate(zip(flattened_actual, flattened_desired, strict=True)):
        dtype = jnp.result_type(actual_, desired_)
        tols = default_tols(dtype.type, rtol=rtol, atol=atol)
        try:
            np.testing.assert_allclose(actual_, desired_, **tols)
        except AssertionError as exception:
            old_message = exception.args[0].split('\n')
            best_part_of_old_message = "\n".join(old_message[3:6]).replace("Max ", "Maximum ")
            test_string = (get_relative_test_string(actual, original_name, original_value, **tols)
                           if original_name is not None and original_value is not None
                           else get_test_string(actual, **tols))
            test_string = "desired = " + test_string
            # style_config = yapf.style.CreatePEP8Style()
            # style_config['COLUMN_LIMIT'] = column_limit
            # test_string = yapf.yapf_api.FormatCode(test_string, style_config=style_config)[0]
            message = (
                f"\nTree leaves don't match at position {i} with rtol={tols['rtol']} and "
                f"atol={tols['atol']}.\n"
                f"{best_part_of_old_message}\n\n"
                f"Actual: {actual}\nDesired: {desired}\n"
                f"Test string:\n{test_string}")
            raise AssertionError(message) from None


def tree_allclose(actual: PyTree,
                  desired: PyTree,
                  rtol: float | None = None,
                  atol: float | None = None) -> bool:
    """Return whether two pytrees are close.

    Args:
        actual: The actual value.
        desired: The desired value.
        rtol: The relative tolerance of the comparisons in the comparison.
        atol: The absolute tolerance of the comparisons in the comparison.
    """
    def allclose(actual_array: Array, desired_array: Array) -> bool:
        dtype = jnp.result_type(actual_array, desired_array)
        tols = default_tols(dtype.type, rtol=rtol, atol=atol)
        return bool(jnp.allclose(actual_array, desired_array, **tols))

    return tree_reduce(and_, tree_map(allclose, actual, desired), True)


# get test string ----------------------------------------------------------------------------------
# Redefinition typing errors in this file are due to https://github.com/python/mypy/issues/2904.
@singledispatch
def get_test_string(actual: Any, rtol: float, atol: float) -> str:
    """Produce a short string of Python code that produces the actual value.

    Args:
        actual: The actual value that was produced.
        rtol: The relative tolerance of the comparisons in the assertion.
        atol: The absolute tolerance of the comparisons in the assertion.
    """
    return repr(actual)


@get_test_string.register(np.ndarray)
@get_test_string.register(jax.Array)
def _(actual: Array | jax.Array, rtol: float, atol: float) -> str:
    if prod(actual.shape) == 0:
        return f"np.empty({actual.shape}, dtype=np.{actual.dtype})"
    with np.printoptions(formatter={'float_kind': partial(_inexact_number_to_string, rtol=rtol,
                                                          atol=atol),
                                    'complex_kind': partial(_inexact_number_to_string, rtol=rtol,
                                                            atol=atol)}):
        return "np." + repr(np.asarray(actual)).replace(' ]', ']').replace(' ,', ',').replace(
            '  ', ' ').replace('dtype=', 'dtype=np.')


@get_test_string.register
def _(actual: Complex, rtol: float, atol: float) -> str:
    x = float(actual) if isinstance(actual, Real) else complex(actual)
    return _inexact_number_to_string(x, rtol, atol)


@get_test_string.register
def _(actual: Integral, rtol: float, atol: float) -> str:
    return str(actual)


@get_test_string.register(list)
@get_test_string.register(tuple)
def _(actual: list[Any] | tuple[Any], rtol: float, atol: float) -> str:
    is_list = isinstance(actual, list)
    is_named_tuple = not is_list and type(actual).__name__ != 'tuple'
    return ((type(actual).__name__ if is_named_tuple else "")
            + ("[" if is_list else "(")
            + ", ".join(get_test_string(sub_actual, rtol, atol)
                        for i, sub_actual in enumerate(actual))
            + (',' if len(actual) == 1 else '')
            + ("]" if is_list else ")"))


@get_test_string.register(dict)
def _(actual: dict[Any, Any], rtol: float, atol: float) -> str:
    return '{' + ",\n".join(repr(key) + ': ' + get_test_string(sub_actual, rtol, atol)
                            for key, sub_actual in actual.items()) + '}'


# get relative test string -------------------------------------------------------------------------
@singledispatch
def get_relative_test_string(actual: Any,
                             original_name: str,
                             original: Any,
                             rtol: float,
                             atol: float) -> str:
    """Produce code for use in tests based on actual and produced values.

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


@get_relative_test_string.register(np.ndarray)
@get_relative_test_string.register(jax.Array)
def _(actual: Array | jax.Array, original_name: str, original: Any, rtol: float,
      atol: float) -> str:
    with np.printoptions(formatter={'float_kind': partial(_inexact_number_to_string, rtol=rtol,
                                                          atol=atol),
                                    'complex_kind': partial(_inexact_number_to_string, rtol=rtol,
                                                            atol=atol)}):
        return "np." + repr(np.asarray(actual)).replace(' ]', ']').replace(' ,', ',').replace('  ',
                                                                                              ' ')


@get_relative_test_string.register
def _(actual: Complex, original_name: str, original: Any, rtol: float, atol: float) -> str:
    return _inexact_number_to_string(actual, rtol, atol)  # type: ignore[arg-type] # pyright: ignore


@get_relative_test_string.register
def _(actual: Integral, original_name: str, original: Any, rtol: float, atol: float) -> str:
    return str(actual)


@get_relative_test_string.register(list)
@get_relative_test_string.register(tuple)
def _(actual: list[Any] | tuple[Any], original_name: str, original: Any, rtol: float,
      atol: float) -> str:
    is_list = isinstance(actual, list)
    return (("[" if is_list else "(")
            + ", ".join(get_relative_test_string(f"{original_name}[{i}]",
                                                 sub_actual, sub_original,
                                                 rtol, atol)
                        for i, (sub_actual, sub_original) in enumerate(zip(actual, original,
                                                                           strict=True)))
            + ("]" if is_list else ")"))


@get_relative_test_string.register(dict)
def _(actual: dict[Any, Any], original_name: str, original: Any, rtol: float, atol: float) -> str:
    if not isinstance(original, dict):
        raise TypeError

    def relative_string(key: Any, sub_actual: Any) -> str:
        return get_relative_test_string(
            f"{original_name}[{key}]", sub_actual, original[key], rtol, atol)

    return '{' + ",\n".join(repr(key) + ': ' + relative_string(key, sub_actual)
                            for key, sub_actual in actual.items()) + '}'


# Private functions --------------------------------------------------------------------------------
def _float_to_string_with_precision(x: complex, precision: int) -> str:
    with np.printoptions(precision=precision, floatmode='maxprec'):
        return repr(np.asarray(x))[6:-1]


def _inexact_number_to_string(x: complex | np.inexact[Any], rtol: float, atol: float) -> str:
    y: float | complex
    if isinstance(x, Real):  # type: ignore[unreachable]
        y = float(x)  # type: ignore[unreachable]
    elif isinstance(x, Complex):  # type: ignore[unreachable]
        y = complex(x)  # type: ignore[unreachable]
    else:
        raise TypeError
    retval = ""  # type: ignore[unreachable]
    for i in range(20):
        retval = _float_to_string_with_precision(y, i)
        if np.isclose(complex(retval), x, rtol=rtol, atol=atol):
            break
    return retval
