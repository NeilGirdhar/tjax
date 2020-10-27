from typing import Any, Callable, Generic, Tuple, TypeVar, Union, cast

import jax
from jax.tree_util import Partial

__all__ = ['jit', 'custom_jvp', 'custom_vjp']


R = TypeVar('R')
F = TypeVar('F', bound=Callable[..., Any])


def as_sorted_tuple(x: Union[int, Tuple[int, ...]]) -> Tuple[int, ...]:
    if isinstance(x, int):
        return (x,)
    return tuple(sorted(x))


def jit(func: F, **kwargs: Any) -> F:
    """
    This version of jit ensures that abstract methods stay abstract.
    """
    retval = cast(F, jax.jit(func, **kwargs))
    if hasattr(func, "__isabstractmethod__"):
        retval.__isabstractmethod__ = func.__isabstractmethod__  # type: ignore
    return retval


class custom_vjp(Generic[R]):
    """
    This is a shim class over jax.custom_vjp to:

    - allow custom_vjp to be used on methods, and
    - rename nondiff_argnums to static_argnums.
    """
    vjp: jax.custom_vjp

    def __init__(self,
                 fun: Callable[..., R],
                 static_argnums: Union[int, Tuple[int, ...]] = ()):
        """
        Args:
            fun: the function to decorate.
            static_argnums: The indices of the static arguments.
        """
        static_argnums = as_sorted_tuple(static_argnums)
        self.vjp = jax.custom_vjp(fun, nondiff_argnums=static_argnums)  # type: ignore

    def defvjp(self, fwd: Callable[..., Tuple[R, Any]], bwd: Callable[..., Any]) -> None:
        self.vjp.defvjp(fwd, bwd)  # type: ignore

    def __call__(self, *args: Any) -> R:
        return self.vjp(*args)

    def __get__(self, instance: Any, owner: Any = None) -> Callable[..., R]:
        if instance is None:
            return self
        # Create a partial function application corresponding to a bound method.
        return Partial(self, instance)


class custom_jvp(Generic[R]):
    """
    This is a shim class over jax.custom_jvp to:

    - allow custom_vjp to be used on methods, and
    - rename nondiff_argnums to static_argnums.
    """
    def __init__(self,
                 fun: Callable[..., R],
                 static_argnums: Union[int, Tuple[int, ...]] = ()):
        """
        Args:
            fun: the function to decorate.
            static_argnums: The indices of the static arguments.
        """
        static_argnums = as_sorted_tuple(static_argnums)
        self.jvp = jax.custom_jvp(fun, nondiff_argnums=static_argnums)  # type: ignore

    def defjvp(self, jvp: Callable[..., R]) -> None:
        """
        Implement the custom forward pass of the custom derivative.

        Args:
            fwd: The custom forward pass.
        """
        self.jvp.defjvp(jvp)  # type: ignore

    def __call__(self, *args: Any) -> R:
        return self.jvp(*args)

    def __get__(self, instance: Any, owner: Any = None) -> Callable[..., R]:
        if instance is None:
            return self
        # Create a partial function application corresponding to a bound method.
        return Partial(self, instance)
