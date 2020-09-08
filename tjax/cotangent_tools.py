from typing import Tuple, TypeVar

from jax.experimental.host_callback import id_print

from .annotations import PyTree
from .shims import custom_vjp

__all__ = ['copy_cotangent', 'print_cotangent']


X = TypeVar('X', bound=PyTree)


@custom_vjp
def copy_cotangent(x: X, y: X) -> Tuple[X, X]:
    return x, y


def _copy_cotangent_fwd(x: X, y: X) -> Tuple[Tuple[X, X], None]:
    return (x, y), None


def _copy_cotangent_bwd(residuals: None, xy_bar: Tuple[X, X]) -> Tuple[X, X]:
    del residuals
    x_bar, _ = xy_bar
    return x_bar, x_bar


copy_cotangent.defvjp(_copy_cotangent_fwd, _copy_cotangent_bwd)


@custom_vjp
def print_cotangent(x: X) -> X:
    return x


def _print_cotangent_fwd(x: X) -> Tuple[X, None]:
    return x, None


def _print_cotangent_bwd(residuals: None, x_bar: X) -> Tuple[X]:
    del residuals
    x_bar = id_print(x_bar)
    return (x_bar,)


print_cotangent.defvjp(_print_cotangent_fwd, _print_cotangent_bwd)
