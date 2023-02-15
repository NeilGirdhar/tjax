from __future__ import annotations

from typing import Any, Callable, Concatenate, Generic, TypeVar, overload

import jax
from jax.tree_util import Partial
from typing_extensions import ParamSpec, Self

__all__ = ['jit', 'custom_jvp', 'custom_vjp']


R_co = TypeVar('R_co', covariant=True)
F = TypeVar('F', bound=Callable[..., Any])
P = ParamSpec('P')
U = TypeVar("U")


def jit(func: F, **kwargs: Any) -> F:
    """This version of jit ensures that abstract methods stay abstract."""
    retval = jax.jit(func, **kwargs)
    if hasattr(func, "__isabstractmethod__"):
        retval.__isabstractmethod__ = func.__isabstractmethod__  # type: ignore[attr-defined]
    # Fixed by https://github.com/NeilGirdhar/jax/tree/jit_annotation.
    return retval  # type: ignore[return-value]


class custom_vjp(Generic[U, P, R_co]):  # noqa: N801
    """A shim class over jax.custom_vjp.

    It
    - allows custom_vjp to be used on methods, and
    - renames nondiff_argnums to static_argnums.

    Args:
        fun: the function to decorate.
        static_argnums: The indices of the static arguments.
    """

    vjp: jax.custom_vjp[R_co]

    def __init__(self,
                 fun: Callable[Concatenate[U, P], R_co],
                 static_argnums: tuple[int, ...] = ()):
        super().__init__()
        static_argnums = tuple(sorted(static_argnums))
        self.vjp = jax.custom_vjp(fun, nondiff_argnums=static_argnums)

    def defvjp(self,
               fwd: Callable[Concatenate[U, P], tuple[R_co, Any]],
               bwd: Callable[..., Any]) -> None:
        self.vjp.defvjp(fwd, bwd)

    def __call__(self,
                 u: U,
                 /,
                 *args: P.args,
                 **kwargs: P.kwargs) -> R_co:
        return self.vjp(u, *args, **kwargs)

    @overload
    def __get__(self, instance: None, owner: Any = None) -> Self:
        ...

    @overload
    def __get__(self, instance: U, owner: Any = None) -> Callable[P, R_co]:
        ...

    def __get__(self, instance: Any, owner: Any = None) -> Callable[..., R_co]:
        if instance is None:
            return self
        # Create a partial function application corresponding to a bound method.
        return Partial(self, instance)  # type: ignore[no-untyped-call]


class custom_jvp(Generic[U, P, R_co]):  # noqa: N801
    """This is a shim class over jax.custom_jvp to allow custom_vjp to be used on methods.

    Args:
        fun: the function to decorate.
        nondiff_argnums: The indices of the non-differentiated arguments.
    """

    def __init__(self,
                 fun: Callable[Concatenate[U, P], R_co],
                 nondiff_argnums: tuple[int, ...] = ()):
        super().__init__()
        nondiff_argnums = tuple(sorted(nondiff_argnums))
        self.jvp = jax.custom_jvp(fun, nondiff_argnums=nondiff_argnums)

    def defjvp(self, jvp: Callable[Concatenate[U, P], tuple[R_co, R_co]]) -> None:
        """Implement the custom forward pass of the custom derivative.

        Args:
            jvp: The custom forward pass.
        """
        self.jvp.defjvp(jvp)

    def __call__(self, u: U, /, *args: P.args, **kwargs: P.kwargs) -> R_co:
        return self.jvp(u, *args, **kwargs)

    @overload
    def __get__(self, instance: None, owner: Any = None) -> Self:
        ...

    @overload
    def __get__(self, instance: U, owner: Any = None) -> Callable[P, R_co]:
        ...

    def __get__(self, instance: Any, owner: Any = None) -> Callable[..., R_co]:
        if instance is None:
            return self
        # Create a partial function application corresponding to a bound method.
        return Partial(self, instance)  # type: ignore[no-untyped-call]
