from __future__ import annotations

import operator
from collections.abc import Callable
from functools import partial
from typing import Any, TypeVar, cast

import jax.numpy as jnp
from jax import tree, vjp
from jax.custom_derivatives import zero_from_primal as jax_zero_from_primal

from .annotations import JaxRealArray, RealNumeric
from .display.print_generic import print_generic
from .shims import custom_jvp, custom_vjp

X = TypeVar('X')
XT = TypeVar('XT', bound=tuple[Any, ...])
Y = TypeVar('Y')


def zero_from_primal(x: X, /, *, symbolic_zeros: bool = False) -> X:
    return jax_zero_from_primal(x, symbolic_zeros=symbolic_zeros)


# scale_cotangent ----------------------------------------------------------------------------------
def scale_cotangent(x: X,
                    scalar_scale: RealNumeric | None = None,
                    tree_scale: X | None = None,
                    ) -> X:
    """Scale x's cotangent.

    Args:
        x: The principal value.
        scalar_scale: A scalar that scales x's cotangent.
        tree_scale: A tree matching x's shape that scales x's cotangent.
    """
    return x


def _scale_cotangent_jvp(scalar_scale: RealNumeric | None,
                         tree_scale: X | None,
                         primals: tuple[X],
                         tangents: tuple[X]
                         ) -> tuple[X, X]:
    x, = primals
    x_dot, = tangents
    if scalar_scale is not None:
        x_dot = tree.map(lambda x_dot_i: x_dot_i * scalar_scale, x_dot)
    if tree_scale is not None:
        x_dot = tree.map(operator.mul, x_dot, tree_scale)
    return x, x_dot


scale_cotangent = custom_jvp(scale_cotangent, nondiff_argnums=(1, 2))
scale_cotangent.defjvp(_scale_cotangent_jvp)


# reverse_scale_cotangent --------------------------------------------------------------------------
@custom_vjp
def reverse_scale_cotangent(x: X) -> tuple[X, JaxRealArray]:
    """Output x and a dummy zero; scale x's cotangent by the zero's cotangent."""
    return x, jnp.zeros(())


def _reverse_scale_cotangent_fwd(x: X) -> tuple[tuple[X, JaxRealArray], None]:
    return (x, jnp.zeros(())), None


def _reverse_scale_cotangent_bwd(residuals: None, xy_bar: tuple[X, JaxRealArray]) -> tuple[X]:
    del residuals
    x_bar, y_bar = xy_bar
    return (tree.map(lambda x_bar_i: x_bar_i * y_bar, x_bar),)


reverse_scale_cotangent.defvjp(_reverse_scale_cotangent_fwd, _reverse_scale_cotangent_bwd)


# replace_cotangent --------------------------------------------------------------------------------
@custom_vjp
def replace_cotangent(x: X, new_cotangent: X) -> X:
    """Set x's cotangent to be new_cotangent's primal value."""
    assert tree.structure(x) == tree.structure(new_cotangent)
    return x


def _replace_cotangent_fwd(x: X, new_cotangent: X) -> tuple[X, X]:
    assert tree.structure(x) == tree.structure(new_cotangent)
    return x, new_cotangent


def _replace_cotangent_bwd(residuals: X, x_bar: X) -> tuple[X, X]:
    assert tree.structure(residuals) == tree.structure(x_bar)
    return residuals, x_bar


replace_cotangent.defvjp(_replace_cotangent_fwd, _replace_cotangent_bwd)


# copy_cotangent -----------------------------------------------------------------------------------
@custom_vjp
def copy_cotangent(x: X, y: X) -> X:
    """Output x, and copy its cotangent to y."""
    assert tree.structure(x) == tree.structure(y)
    return x


def _copy_cotangent_fwd(x: X, y: X) -> tuple[X, None]:
    assert tree.structure(x) == tree.structure(y)
    return x, None


def _copy_cotangent_bwd(residuals: None, x_bar: X) -> tuple[X, X]:
    del residuals
    return x_bar, x_bar


copy_cotangent.defvjp(_copy_cotangent_fwd, _copy_cotangent_bwd)


# print_cotangent ----------------------------------------------------------------------------------
@partial(custom_vjp, static_argnums=(1,))
def print_cotangent(u: X, name: str | None = None) -> X:
    """Print the cotangent of u."""
    return u


def _print_cotangent_fwd(u: X, name: str | None) -> tuple[X, None]:
    return u, None


def _print_cotangent_bwd(name: str | None, residuals: None, x_bar: X) -> tuple[X]:
    del residuals
    if name is None:
        print_generic(x_bar)
    else:
        print_generic(**{name: x_bar})  # type: ignore[arg-type] # pyright: ignore
    return (x_bar,)


# https://github.com/python/mypy/issues/14802
print_cotangent.defvjp(_print_cotangent_fwd,  # type: ignore[arg-type]
                       _print_cotangent_bwd)


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
        this_xs_bar = cast('XT', (xs_zero[:i]
                                  + (x_bar,)
                                  + xs_zero[i + 1:]))
        this_result_bar = (this_xs_bar, scaled_y_bar)
        this_args_bar = f_vjp(this_result_bar)
        all_args_bar.append(this_args_bar)
    return (tuple(all_args_bar),)


cotangent_combinator.defvjp(_cotangent_combinator_fwd, _cotangent_combinator_bwd)
