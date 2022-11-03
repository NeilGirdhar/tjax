from __future__ import annotations

from functools import partial
from typing import Any, Callable, Optional, Tuple, TypeVar, cast

import jax.numpy as jnp
from jax import custom_vjp, vjp
from jax.tree_util import tree_map

from .display import id_display

__all__ = ['copy_cotangent', 'replace_cotangent', 'print_cotangent', 'cotangent_combinator']


X = TypeVar('X')
XT = TypeVar('XT', bound=Tuple[Any, ...])
Y = TypeVar('Y')


# copy_cotangent -----------------------------------------------------------------------------------
@custom_vjp
def copy_cotangent(x: X, y: X) -> X:
    return x


def _copy_cotangent_fwd(x: X, y: X) -> Tuple[X, None]:
    return x, None


def _copy_cotangent_bwd(residuals: None, x_bar: X) -> Tuple[X, X]:
    del residuals
    return x_bar, x_bar


# Pyright can't infer types because custom_vjp doesn't yet depend on ParamSpec.
copy_cotangent.defvjp(_copy_cotangent_fwd, _copy_cotangent_bwd)


# replace_cotangent --------------------------------------------------------------------------------
@custom_vjp
def replace_cotangent(x: X, new_cotangent: X) -> X:
    return x


def _replace_cotangent_fwd(x: X, new_cotangent: X) -> Tuple[X, X]:
    return x, new_cotangent


def _replace_cotangent_bwd(residuals: X, x_bar: X) -> Tuple[X, X]:
    return residuals, x_bar


# Pyright can't infer types because custom_vjp doesn't yet depend on ParamSpec.
replace_cotangent.defvjp(_replace_cotangent_fwd, _replace_cotangent_bwd)


# print_cotangent ----------------------------------------------------------------------------------
@partial(custom_vjp, nondiff_argnums=(1,))
def print_cotangent(u: X, name: Optional[str] = None) -> X:
    return u


def _print_cotangent_fwd(u: X, name: Optional[str]) -> Tuple[X, None]:
    return u, None


def _print_cotangent_bwd(name: Optional[str], residuals: None, x_bar: X) -> Tuple[X]:
    del residuals
    x_bar = id_display(x_bar, name)
    return (x_bar,)


print_cotangent.defvjp(_print_cotangent_fwd, _print_cotangent_bwd)


# cotangent_combinator -----------------------------------------------------------------------------
@partial(custom_vjp, nondiff_argnums=(0,))
def cotangent_combinator(f: Callable[..., Tuple[XT, Y]],
                         args_tuples: Tuple[Tuple[Any, ...], ...]) -> Tuple[XT, Y]:
    """
    Args:
        f: A function that accepts positional arguments and returns xs, y where xs is a tuple of
            length n.
        args_tuples: n copies of the same tuple of positional arguments accepted by f.
    Returns: The pair (xs, y).

    The purpose of the cotangent combinator is to take cotangents of each of the elements of x, and
    send them back through to each of the corresponding argument tuples.
    """
    return f(*args_tuples[0])


def _cotangent_combinator_fwd(f: Callable[..., Tuple[XT, Y]],
                              args_tuples: Tuple[Tuple[Any, ...], ...]
                              ) -> Tuple[Tuple[XT, Y],
                                         Callable[[Tuple[XT, Y]], Tuple[Any, ...]]]:
    return vjp(f, *args_tuples[0])


def _cotangent_combinator_bwd(f: Callable[..., Tuple[XT, Y]],
                              f_vjp: Callable[[Tuple[XT, Y]], Tuple[Any, ...]],
                              xy_bar: Tuple[XT, Y]
                              ) -> Tuple[Any, ...]:
    xs_bar, y_bar = xy_bar
    xs_zero = tuple(tree_map(jnp.zeros_like, x_bar)
                    for x_bar in xs_bar)
    all_args_bar = []
    for i, x_bar in enumerate(xs_bar):
        this_xs_bar = cast(XT, (xs_zero[:i]
                                + (x_bar,)
                                + xs_zero[i + 1:]))
        this_result_bar = (this_xs_bar, y_bar)
        this_args_bar = f_vjp(this_result_bar)
        all_args_bar.append(this_args_bar)
    return (tuple(all_args_bar),)


cotangent_combinator.defvjp(_cotangent_combinator_fwd,  # type: ignore[attr-defined]
                            _cotangent_combinator_bwd)
