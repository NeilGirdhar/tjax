from functools import partial
from typing import Optional, Tuple, TypeVar

from jax import numpy as jnp
from jax.experimental.host_callback import id_print
from jax.tree_util import tree_map

from .annotations import PyTree
from .shims import custom_vjp

__all__ = ['zero_out_cotangent', 'copy_cotangent', 'print_cotangent']


X = TypeVar('X', bound=PyTree)


# zero_out_cotangent -------------------------------------------------------------------------------
@custom_vjp
def zero_out_cotangent(x: X) -> X:
    return x


def _zero_out_cotangent_fwd(x: X) -> Tuple[X, None]:
    return x, None


def _zero_out_cotangent_bwd(residuals: None, x_bar: X) -> X:
    del residuals
    return (tree_map(jnp.zeros_like, x_bar),)


zero_out_cotangent.defvjp(_zero_out_cotangent_fwd, _zero_out_cotangent_bwd)


# copy_cotangent -----------------------------------------------------------------------------------
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


# print_cotangent ----------------------------------------------------------------------------------
@partial(custom_vjp, static_argnums=1)
def print_cotangent(x: X, what: Optional[str] = None) -> X:
    return x


def _print_cotangent_fwd(x: X, what: Optional[str]) -> Tuple[X, None]:
    return x, None


def _print_cotangent_bwd(what: Optional[str], residuals: None, x_bar: X) -> Tuple[X]:
    del residuals
    x_bar = (id_print(x_bar)
             if what is None
             else id_print(x_bar, what=what))
    return (x_bar,)


print_cotangent.defvjp(_print_cotangent_fwd, _print_cotangent_bwd)
