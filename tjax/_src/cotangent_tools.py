from __future__ import annotations

from typing import Optional, Tuple, TypeVar

from .display import id_display
from .shims import custom_vjp

__all__ = ['copy_cotangent', 'replace_cotangent', 'print_cotangent']


X = TypeVar('X')


# copy_cotangent -----------------------------------------------------------------------------------
def copy_cotangent(x: X, y: X) -> Tuple[X, X]:
    return x, y


# Apply after to work around mypy deficiency.
copy_cotangent = custom_vjp(copy_cotangent)  # type: ignore


def _copy_cotangent_fwd(x: X, y: X) -> Tuple[Tuple[X, X], None]:
    return (x, y), None


def _copy_cotangent_bwd(residuals: None, xy_bar: Tuple[X, X]) -> Tuple[X, X]:
    del residuals
    x_bar, _ = xy_bar
    return x_bar, x_bar


copy_cotangent.defvjp(_copy_cotangent_fwd, _copy_cotangent_bwd)  # type: ignore


# replace_cotangent --------------------------------------------------------------------------------
def replace_cotangent(x: X, new_cotangent: X) -> X:
    return x


# Apply after to work around mypy deficiency.
replace_cotangent = custom_vjp(replace_cotangent)  # type: ignore


def _replace_cotangent_fwd(x: X, new_cotangent: X) -> Tuple[X, X]:
    return x, new_cotangent


def _replace_cotangent_bwd(residuals: X, x_bar: X) -> Tuple[X, X]:
    return residuals, x_bar


replace_cotangent.defvjp(_replace_cotangent_fwd, _replace_cotangent_bwd)  # type: ignore


# print_cotangent ----------------------------------------------------------------------------------
def print_cotangent(x: X, name: Optional[str] = None) -> X:
    return x


def _print_cotangent_fwd(x: X, name: Optional[str]) -> Tuple[X, None]:
    return x, None


def _print_cotangent_bwd(name: Optional[str], residuals: None, x_bar: X) -> Tuple[X]:
    del residuals
    x_bar = id_display(x_bar, name)
    return (x_bar,)


# Apply after to work around mypy deficiency.
print_cotangent = custom_vjp(print_cotangent, static_argnums=1)  # type: ignore


print_cotangent.defvjp(_print_cotangent_fwd, _print_cotangent_bwd)  # type: ignore
