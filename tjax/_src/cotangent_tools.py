from __future__ import annotations

from functools import partial

from jax import tree

from .annotations import RealNumeric
from .display.print_generic import print_generic
from .shims import custom_jvp, custom_vjp
from .tree_tools import scale_tree


# scale_cotangent ----------------------------------------------------------------------------------
@partial(custom_jvp, nondiff_argnums=(1, 2))
def scale_cotangent[X](
    x: X,
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


def _scale_cotangent_jvp[X](
    scalar_scale: RealNumeric | None, tree_scale: X | None, primals: tuple[X], tangents: tuple[X]
) -> tuple[X, X]:
    (x,) = primals
    (x_dot,) = tangents
    return x, scale_tree(x_dot, scalar_scale=scalar_scale, tree_scale=tree_scale)


scale_cotangent.defjvp(_scale_cotangent_jvp)


# negate_cotangent ---------------------------------------------------------------------------------
def negate_cotangent[X](x: X) -> X:
    """Return ``x`` while negating the cotangent sent back to ``x``."""
    return scale_cotangent(x, scalar_scale=-1)


# print_cotangent ----------------------------------------------------------------------------------
@partial(custom_vjp, static_argnums=(1,))
def print_cotangent[X](u: X, name: str | None = None) -> X:
    """Print the cotangent of ``u`` during the backward pass and return ``u`` unchanged.

    Args:
        u: The value whose cotangent will be printed.
        name: Optional label passed as a keyword argument to :func:`print_generic`.
            When ``None`` the cotangent is printed positionally.
    """
    return u


def _print_cotangent_fwd[X](u: X, name: str | None) -> tuple[X, None]:
    return u, None


def _print_cotangent_bwd[X](name: str | None, residuals: None, x_bar: X) -> tuple[X]:
    del residuals
    if name is None:
        print_generic(x_bar)
    else:
        print_generic({name: x_bar})
    return (x_bar,)


print_cotangent.defvjp(_print_cotangent_fwd, _print_cotangent_bwd)


# replace_cotangent --------------------------------------------------------------------------------
@custom_vjp
def replace_cotangent[X](x: X, new_cotangent: X) -> X:
    """Set x's cotangent to be new_cotangent's primal value."""
    assert tree.structure(x) == tree.structure(new_cotangent)
    return x


def _replace_cotangent_fwd[X](x: X, new_cotangent: X) -> tuple[X, X]:
    assert tree.structure(x) == tree.structure(new_cotangent)
    return x, new_cotangent


def _replace_cotangent_bwd[X](residuals: X, x_bar: X) -> tuple[X, X]:
    assert tree.structure(residuals) == tree.structure(x_bar)
    return residuals, x_bar


replace_cotangent.defvjp(_replace_cotangent_fwd, _replace_cotangent_bwd)


# copy_cotangent -----------------------------------------------------------------------------------
@custom_vjp
def copy_cotangent[X](x: X, y: X) -> X:
    """Output x, and copy its cotangent to y."""
    assert tree.structure(x) == tree.structure(y)
    return x


def _copy_cotangent_fwd[X](x: X, y: X) -> tuple[X, None]:
    assert tree.structure(x) == tree.structure(y)
    return x, None


def _copy_cotangent_bwd[X](residuals: None, x_bar: X) -> tuple[X, X]:
    del residuals
    return x_bar, x_bar


copy_cotangent.defvjp(_copy_cotangent_fwd, _copy_cotangent_bwd)
