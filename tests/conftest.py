from collections.abc import Generator
from io import StringIO

import numpy as np
import pytest
from jax.experimental import enable_x64
from rich.console import Console


@pytest.fixture(autouse=True)
def _jax_enable64() -> Generator[None]:  # pyright: ignore
    with enable_x64():
        yield


@pytest.fixture
def generator() -> np.random.Generator:
    return np.random.default_rng(123)


@pytest.fixture
def console() -> Console:
    return Console(file=StringIO(), width=80)
