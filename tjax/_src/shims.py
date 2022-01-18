from __future__ import annotations

from typing import Any, Callable, Generic, Tuple, TypeVar

import jax
from jax.tree_util import Partial

__all__ = ['jit', 'custom_jvp', 'custom_vjp']


R = TypeVar('R')
F = TypeVar('F', bound=Callable[..., Any])


def jit(func: F, **kwargs: Any) -> F:
    """
    This version of jit ensures that abstract methods stay abstract.
    """
    retval = jax.jit(func, **kwargs)
    if hasattr(func, "__isabstractmethod__"):
        retval.__isabstractmethod__ = func.__isabstractmethod__  # type: ignore[attr-defined]
    return retval


class custom_vjp(Generic[R]):
    """
    This is a shim class over jax.custom_vjp to:

    - allow custom_vjp to be used on methods, and
    - rename nondiff_argnums to static_argnums.
    """
    vjp: jax.custom_vjp[R]

    def __init__(self,
                 fun: Callable[..., R],
                 static_argnums: Tuple[int, ...] = ()):
        """
        Args:
            fun: the function to decorate.
            static_argnums: The indices of the static arguments.
        """
        super().__init__()
        static_argnums = tuple(sorted(static_argnums))
        self.vjp = jax.custom_vjp(fun, nondiff_argnums=static_argnums)

    def defvjp(self, fwd: Callable[..., Tuple[R, Any]], bwd: Callable[..., Any]) -> None:
        self.vjp.defvjp(fwd, bwd)

    def __call__(self, *args: Any, **kwargs: Any) -> R:
        return self.vjp(*args, **kwargs)

    def __get__(self, instance: Any, owner: Any = None) -> Callable[..., R]:
        if instance is None:
            return self
        # Create a partial function application corresponding to a bound method.
        return Partial(self, instance)  # type: ignore[no-untyped-call]


class custom_jvp(Generic[R]):
    """
    This is a shim class over jax.custom_jvp to:

    - allow custom_vjp to be used on methods, and
    - rename nondiff_argnums to static_argnums.
    """
    def __init__(self,
                 fun: Callable[..., R],
                 static_argnums: Tuple[int, ...] = ()):
        """
        Args:
            fun: the function to decorate.
            static_argnums: The indices of the static arguments.
        """
        super().__init__()
        static_argnums = tuple(sorted(static_argnums))
        self.jvp = jax.custom_jvp(fun, nondiff_argnums=static_argnums)

    def defjvp(self, jvp: Callable[..., Tuple[R, R]]) -> None:
        """
        Implement the custom forward pass of the custom derivative.

        Args:
            fwd: The custom forward pass.
        """
        self.jvp.defjvp(jvp)

    def __call__(self, *args: Any) -> R:
        return self.jvp(*args)

    def __get__(self, instance: Any, owner: Any = None) -> Callable[..., R]:
        if instance is None:
            return self
        # Create a partial function application corresponding to a bound method.
        return Partial(self, instance)  # type: ignore[no-untyped-call]
