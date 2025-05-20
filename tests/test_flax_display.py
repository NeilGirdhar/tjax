from io import StringIO

import jax.numpy as jnp
import pytest
from pytest import CaptureFixture
from rich.console import Console

from tjax import print_generic
from tjax.dataclasses import field

from .test_display import verify

try:
    from flax import nnx

    from tjax.dataclasses import DataClassModule
except ImportError:
    pytest.skip("Skipping NNX display tests", allow_module_level=True)


def test_module_display(capsys: CaptureFixture[str],
                        console: Console) -> None:
    class C(nnx.Module):
        def __init__(self) -> None:
            super().__init__()
            self.x = jnp.zeros(4)
            self.y = 'abc'
            self.a = nnx.Variable(jnp.zeros(4))

    c = C()
    print_generic(c, console=console, immediate=True)
    assert isinstance(console.file, StringIO)
    captured = console.file.getvalue()
    verify(captured,
           """
           C[flax-module]
           ├── x=Jax Array (4,) float64
           │   └──  0.0000 │ 0.0000 │ 0.0000 │ 0.0000
           ├── y="abc"
           └── a=Variable
               └── value=Jax Array (4,) float64
                   └──  0.0000 │ 0.0000 │ 0.0000 │ 0.0000
           """)


def test_dataclass_module_display(capsys: CaptureFixture[str],
                                  console: Console) -> None:
    class C(DataClassModule):
        x: float
        y: str = field(static=True)

        def __post_init__(self, rngs: nnx.Rngs) -> None:
            if hasattr(super(), '__post_init__'):
                super().__post_init__(rngs)
            self.a = nnx.Variable(jnp.zeros(4))

    c = C(0.5, 'abc', rngs=nnx.Rngs())
    print_generic(c, console=console)
    assert isinstance(console.file, StringIO)
    captured = console.file.getvalue()
    verify(captured,
           """
           C[dataclass,flax-module]
           ├── a=Variable
           │   └── value=Jax Array (4,) float64
           │       └──  0.0000 │ 0.0000 │ 0.0000 │ 0.0000
           ├── x=0.5
           └── y="abc"
           """)
