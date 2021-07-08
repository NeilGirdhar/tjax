import re

import pytest
from pytest import CaptureFixture

from tjax import assert_tree_allclose


@pytest.mark.parametrize("actual, desired",
                         [(0.1234567890123, "0\\.123457"),
                          (0.000123456789, "0\\.00012346"),
                          (1.23456789e-8, "1\\.e-08"),
                          (1.23456789e8, "1\\.23457e\\+08")])
def test_testing(capsys: CaptureFixture[str], actual: float, desired: str) -> None:
    with pytest.raises(AssertionError) as exception:
        assert_tree_allclose(actual, 1e6, rtol=1e-5, atol=1e-8)
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""  # Nothing should be printed.
    assertion_string = exception.value.args[0]
    pattern = rf"""
JAX trees don't match with rtol=1e-05 and atol=1e-08\.
Mismatched elements: 1 / 1 \(100%\)
Maximum absolute difference: .+
Maximum relative difference: .+
Actual: {actual}
Desired: 1000000\.0
Test string:
actual = {desired}
"""
    assert re.match(pattern, assertion_string)
