from typing import Any

import pytest
from _pytest.capture import CaptureFixture

from tjax import assert_jax_allclose


@pytest.mark.parametrize("actual, desired",
                         [(0.1234567890123, "0.123457"),
                          (0.000123456789, "0.00012346"),
                          (1.23456789e-8, "1.e-08"),
                          (1.23456789e8, "1.23457e+08")])
def test_testing(capsys: CaptureFixture, actual: float, desired: str) -> None:
    with pytest.raises(AssertionError):
        assert_jax_allclose(actual, 1e6, "", 0.0, rtol=1e-5, atol=1e-8)
    captured: Any = capsys.readouterr()  # type: ignore
    assert captured.out == f"""JAX trees don't match.  Actual:
{actual}
Desired:
1000000.0
Test string:
{desired}
"""
