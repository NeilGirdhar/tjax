from collections.abc import Generator

import numpy as np
import pytest
from jax.experimental import enable_x64
from rich.console import Console


@pytest.fixture(autouse=True)
def _jax_enable64() -> Generator[None, None, None]:
    with enable_x64():
        yield


@pytest.fixture()
def generator() -> np.random.Generator:
    return np.random.default_rng(123)


@pytest.fixture()
def console() -> Console:
    return Console(no_color=True,
                   width=80,
                   height=60,
                   force_terminal=True)
