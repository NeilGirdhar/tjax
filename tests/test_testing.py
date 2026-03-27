from __future__ import annotations

import re

import pytest

from tjax import assert_tree_allclose, tree_allclose
from tjax.dataclasses import dataclass


@pytest.mark.parametrize(
    ("actual", "desired"),
    [
        (0.1234567890123, "0\\.123457"),
        (0.000123456789, "0\\.00012346"),
        (1.23456789e-8, "1\\.e-08"),
        (1.23456789e8, "1\\.23457e\\+08"),
    ],
)
def test_testing(capsys: pytest.CaptureFixture[str], actual: float, desired: str) -> None:
    with pytest.raises(AssertionError) as exception:
        assert_tree_allclose(actual, 1e6, rtol=1e-5, atol=1e-8)
    captured = capsys.readouterr()
    assert not captured.out  # Nothing should be printed.
    assert not captured.err
    assertion_string = exception.value.args[0]
    pattern = rf"""
Tree leaves don't match at position 0 with rtol=1e-05 and atol=1e-08\.
Mismatched elements: 1 / 1 \(100%\)
Maximum absolute difference among violations: .+
Maximum relative difference among violations: .+

Actual: {actual}
Desired: 1000000\.0
Test string:
desired = {desired}"""
    assert re.match(pattern, assertion_string)


@pytest.mark.parametrize(
    ("actual", "desired", "result"),
    [
        ((), (), True),
        ((1.2, 3.4), (1.2, 3.4), True),
        ({"a": 1.2}, {"a": 1.2}, True),
        (1.3, 1.4, False),
    ],
)
def test_tree_allclose(actual: object, desired: object, *, result: bool | None) -> None:
    assert tree_allclose(actual, desired) == result


def test_tree_allclose_structure_mismatch() -> None:
    with pytest.raises(ValueError, match="Tuple arity mismatch"):
        tree_allclose((1, 2), (1, 2, 3))


def test_assert_tree_allclose_relative_dataclass_message() -> None:
    @dataclass
    class Point:
        x: int
        y: int

    original = Point(1, 2)
    desired = Point(0, 0)
    actual = Point(3, 4)

    with pytest.raises(AssertionError) as exception:
        assert_tree_allclose(actual, desired, "original", original)

    assertion_string = exception.value.args[0]
    assert "desired = replace(original, x=3,\ny=4)" in assertion_string
