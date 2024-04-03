from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any, TypeVar, cast

import jax.numpy as jnp
from jax import tree, vjp

from .annotations import JaxRealArray, RealNumeric
from .display.tapped import tapped_print_generic
from .shims import custom_jvp, custom_vjp

__all__ = [
    'copy_cotangent',
    'cotangent_combinator',
    'print_cotangent',
    'replace_cotangent',
    'scale_cotangent']


X = TypeVar('X')
XT = TypeVar('XT', bound=tuple[Any, ...])
Y = TypeVar('Y')


# scale_cotangent ----------------------------------------------------------------------------------
def scale_cotangent(x: X, scale: RealNumeric) -> X:
    return x


def _scale_cotangent_jvp(scale: RealNumeric, primals: tuple[X], tangents: tuple[X]) -> tuple[X, X]:
    x, = primals
    x_dot, = tangents
    scaled_x_dot = tree.map(lambda x_dot_i: x_dot_i * scale, x_dot)
    return x, scaled_x_dot


scale_cotangent = custom_jvp(scale_cotangent, nondiff_argnums=(1,))
# The type variables don't match, so this type error needs to be ignored.
scale_cotangent.defjvp(_scale_cotangent_jvp)  # pyright: ignore


# copy_cotangent -----------------------------------------------------------------------------------
@custom_vjp
def copy_cotangent(x: X, y: X) -> X:
    assert tree.structure(x) == tree.structure(y)
    return x


def _copy_cotangent_fwd(x: X, y: X) -> tuple[X, None]:
    assert tree.structure(x) == tree.structure(y)
    return x, None


def _copy_cotangent_bwd(residuals: None, x_bar: X) -> tuple[X, X]:
    del residuals
    return x_bar, x_bar


# Pyright can't infer types because custom_vjp doesn't yet depend on ParamSpec.
copy_cotangent.defvjp(_copy_cotangent_fwd, _copy_cotangent_bwd)


# replace_cotangent --------------------------------------------------------------------------------
@custom_vjp
def replace_cotangent(x: X, new_cotangent: X) -> X:
    assert tree.structure(x) == tree.structure(new_cotangent)
    return x


def _replace_cotangent_fwd(x: X, new_cotangent: X) -> tuple[X, X]:
    assert tree.structure(x) == tree.structure(new_cotangent)
    return x, new_cotangent


def _replace_cotangent_bwd(residuals: X, x_bar: X) -> tuple[X, X]:
    assert tree.structure(residuals) == tree.structure(x_bar)
    return residuals, x_bar


# Pyright can't infer types because custom_vjp doesn't yet depend on ParamSpec.
replace_cotangent.defvjp(_replace_cotangent_fwd, _replace_cotangent_bwd)


# print_cotangent ----------------------------------------------------------------------------------
@partial(custom_vjp, static_argnums=(1,))
def print_cotangent(u: X, name: str | None = None) -> X:
    return u


def _print_cotangent_fwd(u: X, name: str | None) -> tuple[X, None]:
    return u, None


def _print_cotangent_bwd(name: str | None, residuals: None, x_bar: X) -> tuple[X]:
    del residuals
    x_bar = (tapped_print_generic(x_bar)
        if name is None
        else tapped_print_generic(
            **{name: x_bar}))  # type: ignore[call-overload] # pyright: ignore
    return (x_bar,)


# https://github.com/python/mypy/issues/14802
print_cotangent.defvjp(_print_cotangent_fwd, _print_cotangent_bwd)


# cotangent_combinator -----------------------------------------------------------------------------
def cotangent_combinator(f: Callable[..., tuple[XT, Y]],
                         args_tuples: tuple[tuple[Any, ...], ...],
                         aux_cotangent_scales: tuple[RealNumeric, ...] | None) -> tuple[XT, Y]:
    """Run a function once, but send differerent cotangents back to each input.

    Args:
        f: A function that accepts positional arguments and returns xs, y where xs is a tuple of
            length n.
        args_tuples: n copies of the same tuple of positional arguments accepted by f.
        aux_cotangent_scales: If provided, scale each of the cotangents.
    Returns: The pair (xs, y).

    The purpose of the cotangent combinator is to take cotangents of each of the elements of x, and
    send them back through to each of the corresponding argument tuples.
    """
    return f(*args_tuples[0])


cotangent_combinator = custom_vjp(cotangent_combinator, static_argnums=(0, 2))


def _cotangent_combinator_fwd(f: Callable[..., tuple[XT, Y]],
                              args_tuples: tuple[tuple[Any, ...], ...],
                              aux_cotangent_scales: tuple[RealNumeric, ...] | None
                              ) -> tuple[tuple[XT, Y],
                                         Callable[[tuple[XT, Y]], tuple[Any, ...]]]:
    return vjp(f, *args_tuples[0])


def _cotangent_combinator_bwd(f: Callable[..., tuple[XT, Y]],
                              aux_cotangent_scales: tuple[RealNumeric, ...] | None,
                              f_vjp: Callable[[tuple[XT, Y]], tuple[Any, ...]],
                              xy_bar: tuple[XT, Y]
                              ) -> tuple[Any, ...]:
    xs_bar, y_bar = xy_bar
    if aux_cotangent_scales is None:
        aux_cotangent_scales = tuple(1.0 for _ in xs_bar)
    xs_zero = tuple(tree.map(jnp.zeros_like, x_bar)
                    for x_bar in xs_bar)
    all_args_bar = []
    for i, (x_bar, aux_cotangent_scale) in enumerate(zip(xs_bar, aux_cotangent_scales,
                                                         strict=True)):
        scaled_y_bar = tree.map(lambda y_bar_i, scale=aux_cotangent_scale: y_bar_i * scale,
                                y_bar)
        this_xs_bar = cast(XT, (xs_zero[:i]
                                + (x_bar,)
                                + xs_zero[i + 1:]))
        this_result_bar = (this_xs_bar, scaled_y_bar)
        this_args_bar = f_vjp(this_result_bar)
        all_args_bar.append(this_args_bar)
    return (tuple(all_args_bar),)


cotangent_combinator.defvjp(_cotangent_combinator_fwd, _cotangent_combinator_bwd)


# augment_cotangent --------------------------------------------------------------------------------
@custom_vjp
def augment_cotangent(loss: JaxRealArray,
                      x: X,
                      cotangent_addend: X
                      ) -> JaxRealArray:
    """Create a cotangent for x equal to the cotangent_addend times the loss cotangent."""
    assert tree.structure(x) == tree.structure(cotangent_addend)
    return loss


# Private functions --------------------------------------------------------------------------------
def _augment_cotangent_fwd(loss: JaxRealArray,
                           x: X,
                           cotangent_addend: X
                           ) -> tuple[JaxRealArray, X]:
    assert tree.structure(x) == tree.structure(cotangent_addend)
    return loss, cotangent_addend


def _augment_cotangent_bwd(cotangent_addend: X,
                           loss_bar: JaxRealArray
                           ) -> tuple[None, X, None]:
    x_bar = tree.map(lambda x: x * loss_bar, cotangent_addend)
    return (None, x_bar, None)


augment_cotangent.defvjp(_augment_cotangent_fwd, _augment_cotangent_bwd)
