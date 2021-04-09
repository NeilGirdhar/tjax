from collections.abc import Generator

import numpy as np
import pytest
from jax.experimental import enable_x64


@pytest.fixture(autouse=True)
def _jax_enable64() -> Generator[None, None, None]:
    with enable_x64():
        yield


@pytest.fixture
def generator() -> np.random.Generator:
    return np.random.default_rng(123)
