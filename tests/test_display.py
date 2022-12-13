import colorful as cf
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from pytest import CaptureFixture

from tjax import RealArray, print_generic, tapped_print_generic


def test_numpy_display(capsys: CaptureFixture[str]) -> None:
    print_generic(np.ones(3))
    captured = capsys.readouterr()
    assert (captured.out == str(cf.yellow("NumPy Array (3,) float64"))
            + '\n        1.0000      1.0000      1.0000\n\n')


def test_jax_numpy_display(capsys: CaptureFixture[str]) -> None:
    print_generic(jnp.ones(3))
    captured = capsys.readouterr()
    assert (captured.out == str(cf.violet("Jax Array (3,) float64"))
            + '\n        1.0000      1.0000      1.0000\n\n')


def test_jit_display(capsys: CaptureFixture[str]) -> None:
    @jit
    def f(x: RealArray) -> RealArray:
        print_generic(x)
        return x
    f(1.0)
    captured = capsys.readouterr()
    assert captured.out == str(cf.violet("Jax Array () float64")) + '\n\n'


def test_batch_display(capsys: CaptureFixture[str]) -> None:
    @jit
    def f(x: RealArray) -> RealArray:
        print_generic(x)
        return x
    vmap(f)(jnp.ones(10))
    captured = capsys.readouterr()
    assert captured.out == str(cf.magenta("BatchTracer () float64 batched over 10")) + '\n\n'


def test_tapped(capsys: CaptureFixture[str]) -> None:
    "Like test_jit_display, but uses tapped_print_generic to get the array."
    @jit
    def f(x: RealArray) -> RealArray:
        x = tapped_print_generic(x)
        return x
    f(np.ones(3))
    captured = capsys.readouterr()
    assert (captured.out == str(cf.yellow("NumPy Array (3,) float64"))
            + '\n        1.0000      1.0000      1.0000\n\n')


def test_dict(capsys: CaptureFixture[str]) -> None:
    print_generic({'cat': 5,
                   'mouse': {'dog': 3,
                             'sheep': 4}})
    captured = capsys.readouterr()
    output = captured.out
    e = cf.base00('=')
    correct = f"""{cf.orange('dict')}
    {cf.blue('cat')}{e}{cf.cyan('5')}
    {cf.blue('mouse')}{e}{cf.orange('dict')}
        {cf.blue('dog')}{e}{cf.cyan('3')}
        {cf.blue('sheep')}{e}{cf.cyan('4')}

""".replace('\x1b[26m', '')
    assert (output == correct)
