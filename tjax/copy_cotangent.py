from typing import Tuple, TypeVar

from .annotations import PyTree
from .shims import custom_vjp

__all__ = ['copy_cotangent']


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
