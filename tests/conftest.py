import numpy as np
import pytest


@pytest.fixture
def generator() -> np.random.Generator:
    return np.random.default_rng(123)
