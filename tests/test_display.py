from textwrap import dedent

import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from pytest import CaptureFixture
from rich.console import Console

from tjax import RealArray, print_generic, tapped_print_generic
from tjax.dataclasses import dataclass


def verify(actual: str, desired: str) -> None:
    actual = "\n".join(x.rstrip() for x in actual.strip().split('\n'))
    assert actual == dedent(desired).strip()


def test_numpy_display(capsys: CaptureFixture[str],
                       console: Console) -> None:
    print_generic(numpy_array=np.arange(6.0).reshape((3, 2)), console=console)
    captured = capsys.readouterr()
    verify(captured.out,
           """
           numpy_array=NumPy Array (3, 2) float64
           └──      0.0000 │     1.0000
                    2.0000 │     3.0000
                    4.0000 │     5.0000
           """)


def test_numpy_display_big(capsys: CaptureFixture[str],
                           console: Console) -> None:
    print_generic(numpy_array=np.arange(30.0).reshape((15, 2)), console=console)
    captured = capsys.readouterr()
    verify(captured.out,
           """
           numpy_array=NumPy Array (15, 2) float64
           ├── mean=14.5
           └── deviation=8.65544144839919
           """)


def test_jax_numpy_display(capsys: CaptureFixture[str],
                           console: Console) -> None:
    print_generic(jnp.ones(3), console=console)
    captured = capsys.readouterr()
    verify(captured.out,
           """
           Jax Array (3,) float64
           └──      1.0000 │     1.0000 │     1.0000
           """)


def test_jit_display(capsys: CaptureFixture[str],
                     console: Console) -> None:
    @jit
    def f(x: RealArray) -> RealArray:
        print_generic(x, console=console)
        return x
    f(1.0)
    captured = capsys.readouterr()
    verify(captured.out, "Jax Array () float64")


def test_batch_display(capsys: CaptureFixture[str],
                       console: Console) -> None:
    @jit
    def f(x: RealArray) -> RealArray:
        print_generic(x, console=console)
        return x
    vmap(f)(jnp.ones(10))
    captured = capsys.readouterr()
    verify(captured.out, "BatchTracer () float64 batched over 10")


def test_tapped(capsys: CaptureFixture[str],
                console: Console) -> None:
    "Like test_jit_display, but uses tapped_print_generic to get the array."
    @jit
    def f(x: RealArray) -> RealArray:
        x = tapped_print_generic(x, console=console)
        return x
    f(np.ones(3))
    captured = capsys.readouterr()
    verify(captured.out,
           """
           NumPy Array (3,) float64
           └──      1.0000 │     1.0000 │     1.0000
           """)


def test_dict(capsys: CaptureFixture[str],
              console: Console) -> None:
    print_generic({'cat': 5,
                   'mouse': {'dog': 3,
                             'sheep': 4}},
                  console=console)
    captured = capsys.readouterr()
    verify(captured.out,
           """
           dict
           ├── cat=5
           └── mouse=dict
               ├── dog=3
               └── sheep=4
           """)


def test_dataclass(capsys: CaptureFixture[str],
                   console: Console) -> None:
    @dataclass
    class C:
        x: int
        y: int

    @dataclass
    class D:
        c: C
        d: C

    print_generic(d=D(C(1, 2), C(3, 4)),
                  console=console)
    captured = capsys.readouterr()
    verify(captured.out,
           """
           d=D
           ├── c=C
           │   ├── x=1
           │   └── y=2
           └── d=C
               ├── x=3
               └── y=4
           """)