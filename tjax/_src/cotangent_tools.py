from __future__ import annotations

from functools import partial
from typing import Optional, Tuple, TypeVar

from .display import id_display
from .shims import custom_vjp

__all__ = ['copy_cotangent', 'replace_cotangent', 'print_cotangent']


X = TypeVar('X')


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
@partial(custom_vjp, static_argnums=(1,))  # type: ignore[arg-type]
def print_cotangent(u: X, name: Optional[str] = None) -> X:
    return u


def _print_cotangent_fwd(u: X, name: Optional[str]) -> Tuple[X, None]:
    return u, None


def _print_cotangent_bwd(name: Optional[str], residuals: None, x_bar: X) -> Tuple[X]:
    del residuals
    x_bar = id_display(x_bar, name)
    return (x_bar,)


print_cotangent.defvjp(_print_cotangent_fwd, _print_cotangent_bwd)  # type: ignore[arg-type]
