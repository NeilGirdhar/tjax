from typing import Callable, Optional, Tuple, TypeVar

from jax.experimental.host_callback import id_print
from jax.tree_util import tree_map

from .annotations import Array
from .shims import custom_vjp

__all__ = ['transform_cotangent', 'mapped_transform_cotangent', 'copy_cotangent',
           'replace_cotangent', 'print_cotangent']


X = TypeVar('X')


# transform_cotangent ------------------------------------------------------------------------------
def transform_cotangent(f: Callable[[X], X], x: X) -> X:
    return x


# Apply after to work around mypy deficiency.
transform_cotangent = custom_vjp(transform_cotangent, static_argnums=0)  # type: ignore


def _transform_cotangent_fwd(f: Callable[[X], X], x: X) -> Tuple[X, None]:
    return x, None


def _transform_cotangent_bwd(f: Callable[[X], X], residuals: None, x_bar: X) -> Tuple[X]:
    return (f(x_bar),)


transform_cotangent.defvjp(_transform_cotangent_fwd, _transform_cotangent_bwd)  # type: ignore


def mapped_transform_cotangent(g: Callable[[Array], Array], x: X) -> X:
    def f(x_bar: X) -> X:
        return tree_map(g, x_bar)
    return transform_cotangent(f, x)


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


# Apply after to work around mypy deficiency.
print_cotangent = custom_vjp(print_cotangent, static_argnums=1)  # type: ignore


print_cotangent.defvjp(_print_cotangent_fwd, _print_cotangent_bwd)  # type: ignore
