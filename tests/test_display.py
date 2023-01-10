from textwrap import dedent
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from jax import enable_custom_prng, jit, vmap
from jax.random import KeyArray, PRNGKey
from pytest import CaptureFixture
from rich.console import Console

from tjax import RealArray, print_generic, tapped_print_generic
from tjax.dataclasses import dataclass, field


def verify(actual: str, desired: str) -> None:
    actual = "\n".join(x.rstrip() for x in actual.strip().split('\n'))
    assert actual == dedent(desired).strip()


def test_numpy_display(capsys: CaptureFixture[str],
                       console: Console) -> None:
    print_generic(numpy_array=np.reshape(np.arange(6.0), (3, 2)), console=console)
    captured = capsys.readouterr()
    verify(captured.out,
           """
           numpy_array=NumPy Array (3, 2) float64
           └──  0.0000 │ 1.0000
                2.0000 │ 3.0000
                4.0000 │ 5.0000
           """)


def test_numpy_display_big(capsys: CaptureFixture[str],
                           console: Console) -> None:
    print_generic(numpy_array=np.reshape(np.arange(30.0), (15, 2)), console=console)
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
           └──  1.0000 │ 1.0000 │ 1.0000
           """)


def test_jit_display(capsys: CaptureFixture[str],
                     console: Console) -> None:
    @jit
    def f(x: RealArray) -> RealArray:
        print_generic(x, console=console)
        return x
    f(jnp.asarray(1.0))
    captured = capsys.readouterr()
    verify(captured.out, "Jax Array () float64")


def test_batch_display(capsys: CaptureFixture[str],
                       console: Console) -> None:
    @jit
    def f(x: RealArray) -> RealArray:
        print_generic(x, console=console)
        return x
    vmap(vmap(f, in_axes=2), in_axes=1)(jnp.ones((3, 4, 5, 6)))
    captured = capsys.readouterr()
    # Unfortunately, there's no way anymore to detect batch axes:
    # batched over axes of size (4, 6)
    verify(captured.out, "Jax Array (3, 5) float64")


def test_batch_display_dict(capsys: CaptureFixture[str],
                            console: Console) -> None:
    @jit
    def f(x: RealArray) -> RealArray:
        print_generic({'abc': x}, console=console)
        return x
    vmap(vmap(f))(jnp.ones((3, 4)))
    captured = capsys.readouterr()
    # Unfortunately, there's no way anymore to detect batch axes:
    # batched over axes of size (3, 4)
    verify(captured.out,
           """
           dict
           └── abc=Jax Array () float64
           """)


def test_tapped(capsys: CaptureFixture[str],
                console: Console) -> None:
    """Like test_jit_display, but uses tapped_print_generic to get the array."""
    @jit
    def f(x: RealArray) -> RealArray:
        return tapped_print_generic(x, console=console)
    f(np.ones(3))
    captured = capsys.readouterr()
    verify(captured.out,
           """
           NumPy Array (3,) float64
           └──  1.0000 │ 1.0000 │ 1.0000
           """)


def test_tapped_batched(capsys: CaptureFixture[str],
                        console: Console) -> None:
    """Like test_batch_display, but uses tapped_print_generic to get the array."""
    @jit
    def f(x: RealArray) -> RealArray:
        tapped_print_generic(x, console=console)
        return x
    vmap(vmap(f, in_axes=2), in_axes=1)(jnp.ones((3, 4, 5, 6)))
    captured = capsys.readouterr()
    verify(captured.out,
           """
           NumPy Array (3, 5) float64 batched over axes of size (4, 6)
           ├── mean=1.0
           └── deviation=0.0
           """)


def test_tapped_dict(capsys: CaptureFixture[str],
                     console: Console) -> None:
    """Like test_batch_display, but uses tapped_print_generic to get the array."""
    @jit
    def f(x: RealArray) -> RealArray:
        tapped_print_generic(x=x, console=console)
        return x
    vmap(vmap(f, in_axes=2), in_axes=1)(jnp.ones((3, 4, 5, 6)))
    captured = capsys.readouterr()
    verify(captured.out,
           """
           x=NumPy Array (3, 5) float64 batched over axes of size (4, 6)
           ├── mean=1.0
           └── deviation=0.0
           """)


# Unskip when https://github.com/google/jax/issues/13949 is resolved.
@pytest.mark.skip()
def test_tapped_key(capsys: CaptureFixture[str],
                    console: Console) -> None:
    with enable_custom_prng():
        key = PRNGKey(123)
        @jit
        def f(x: KeyArray) -> KeyArray:
            return tapped_print_generic(x)

        f(key)
    captured = capsys.readouterr()
    verify(captured.out,
           """
           x=NumPy Array (3, 5) float64 batched over axes of size (4, 6)
           ├── mean=1.0
           └── deviation=0.0
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


if __name__ == "__main__":
    @dataclass
    class Triplet:
        x: Any
        y: Any
        z: Any = field(static=True)

    a = Triplet(np.reshape(np.arange(6.0), (3, 2)),
                np.reshape(np.arange(30.0), (15, 2)),
                Triplet)

    @jit
    def g(x: RealArray) -> None:
        print_generic(Triplet({'abc': Triplet(a,
                                              x,
                                              2)},
                              a,
                              'blah'))
        tapped_print_generic(Triplet({'abc': Triplet(a,
                                                     x,
                                                     2)},
                                     a,
                                     'blah'))
    vmap(vmap(g, in_axes=2), in_axes=1)(jnp.ones((3, 4, 5, 6)))
