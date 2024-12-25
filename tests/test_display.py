from io import StringIO
from textwrap import dedent
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array, enable_custom_prng, jit, tree, vmap
from jax.random import key
from pytest import CaptureFixture
from rich.console import Console

from tjax import KeyArray, RealArray, print_generic
from tjax.dataclasses import dataclass, field


def verify(actual: str, desired: str) -> None:
    actual = "\n".join(x.rstrip() for x in actual.strip().split('\n'))
    assert actual == dedent(desired).strip()


def test_numpy_display(capsys: CaptureFixture[str],
                       console: Console) -> None:
    print_generic(numpy_array=np.reshape(np.arange(6.0), (3, 2)), console=console)
    assert isinstance(console.file, StringIO)
    captured = console.file.getvalue()
    verify(captured,
           """
           numpy_array=NumPy Array (3, 2) float64
           └──  0.0000 │ 1.0000
                2.0000 │ 3.0000
                4.0000 │ 5.0000
           """)


def test_numpy_display_big(capsys: CaptureFixture[str],
                           console: Console) -> None:
    print_generic(numpy_array=np.reshape(np.arange(30.0), (15, 2)), console=console)
    assert isinstance(console.file, StringIO)
    captured = console.file.getvalue()
    verify(captured,
           """
           numpy_array=NumPy Array (15, 2) float64
           ├── mean=14.5
           └── deviation=8.65544144839919
           """)


def test_jax_numpy_display(capsys: CaptureFixture[str],
                           console: Console) -> None:
    print_generic(jnp.ones(3), console=console)
    assert isinstance(console.file, StringIO)
    captured = console.file.getvalue()
    verify(captured,
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
    assert isinstance(console.file, StringIO)
    captured = console.file.getvalue()
    verify(captured,
           """
           Jax Array () float64
           └── 1.0000
           """)


def test_vmap_display(capsys: CaptureFixture[str],
                      console: Console) -> None:
    """Like test_batch_display, but uses print_generic to get the array."""
    @jit
    def f(x: RealArray) -> RealArray:
        print_generic(x=x, console=console)
        return x
    vmap(vmap(f, in_axes=2), in_axes=1)(jnp.ones((3, 4, 5, 6)))
    assert isinstance(console.file, StringIO)
    captured = console.file.getvalue()
    s = dedent("""
               x=Jax Array (3, 5) float64
               └──  1.0000 │ 1.0000 │ 1.0000 │ 1.0000 │ 1.0000
                    1.0000 │ 1.0000 │ 1.0000 │ 1.0000 │ 1.0000
                    1.0000 │ 1.0000 │ 1.0000 │ 1.0000 │ 1.0000""")
    verify(captured, s * 24)


def test_key_display(capsys: CaptureFixture[str],
                    console: Console) -> None:
    with enable_custom_prng():
        k = key(123)

        @jit
        def f(x: KeyArray) -> None:
            print_generic(x, console=console)

        f(k)
    assert isinstance(console.file, StringIO)
    captured = console.file.getvalue()
    verify(captured,
           """
           Jax Array (2,) uint32
           └──  0 │ 123
           """)


def test_dict_display(capsys: CaptureFixture[str],
                      console: Console) -> None:
    print_generic({'cat': 5,
                   'mouse': {'dog': 3,
                             'sheep': 4}},
                  console=console)
    assert isinstance(console.file, StringIO)
    captured = console.file.getvalue()
    verify(captured,
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
    assert isinstance(console.file, StringIO)
    captured = console.file.getvalue()
    verify(captured,
           """
           d=D[dataclass]
           ├── c=C[dataclass]
           │   ├── x=1
           │   └── y=2
           └── d=C[dataclass]
               ├── x=3
               └── y=4
           """)


def test_pytreedef(capsys: CaptureFixture[str],
                   console: Console) -> None:
    @dataclass
    class C:
        x: int
        y: int

    _, tree_def = tree.flatten(C(1, 2))

    print_generic(tree_def, immediate=True, console=console)
    assert isinstance(console.file, StringIO)
    captured = console.file.getvalue()
    verify(captured,
           f"""
           PyTreeDef
           └── hash={hash(tree_def)}
           """)


def test_seen_array(capsys: CaptureFixture[str],
                    console: Console) -> None:
    @dataclass
    class C:
        x: Array
        y: Array

    z = jnp.zeros(2)
    tree_def = C(z, z)

    print_generic(tree_def, immediate=True, console=console)
    assert isinstance(console.file, StringIO)
    captured = console.file.getvalue()
    verify(captured,
           """
           C[dataclass]
           ├── x=Jax Array (2,) float64
           │   └──  0.0000 │ 0.0000
           └── y=Jax Array (2,) float64
               └──  0.0000 │ 0.0000
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
        print_generic(Triplet({'abc': Triplet(a,
                                                     x,
                                                     2)},
                                     a,
                                     'blah'))
    vmap(vmap(g, in_axes=2), in_axes=1)(jnp.ones((3, 4, 5, 6)))
