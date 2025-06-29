from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import update_wrapper
from typing import Any, Concatenate, Generic, Self, TypeVar, overload

import jax
from jax.tree_util import Partial
from typing_extensions import ParamSpec, override

from .function_markers import all_wrapper_assignments

R_co = TypeVar('R_co', covariant=True)
F = TypeVar('F', bound=Callable[..., Any])
P = ParamSpec('P')
U = TypeVar("U")


def jit(func: F, **kwargs: object) -> F:
    """A version of jax.jit that preserves flags.

    This ensures that abstract methods stay abstract, method overrides remain overrides.
    """
    retval = jax.jit(func, **kwargs)  # pyright: ignore
    _ = update_wrapper(retval, func, all_wrapper_assignments)
    # Return type is fixed by https://github.com/NeilGirdhar/jax/tree/jit_annotation.
    return retval  # type: ignore[return-value] # pyright: ignore


class custom_vjp(Generic[P, R_co]):  # noqa: N801
    """A shim class over jax.custom_vjp that uses ParamSpec.

    Args:
        func: The function to decorate.
        static_argnums: The indices of the **static** arguments -- nothing to do with
            differentiation.
    """

    vjp: jax.custom_vjp[R_co]

    @override
    def __init__(self,
                 func: Callable[P, R_co],
                 *,
                 static_argnums: tuple[int, ...] = ()
                 ) -> None:
        super().__init__()
        static_argnums = tuple(sorted(static_argnums))
        self.vjp = jax.custom_vjp(func, nondiff_argnums=static_argnums)
        _ = update_wrapper(self, func, all_wrapper_assignments)

    def defvjp(self,
               fwd: Callable[P, tuple[R_co, Any]],
               bwd: Callable[..., Any]) -> None:
        self.vjp.defvjp(fwd, bwd)

    def __call__(self, /, *args: P.args, **kwargs: P.kwargs) -> R_co:
        return self.vjp(*args, **kwargs)


class custom_vjp_method(Generic[U, P, R_co]):  # noqa: N801
    """A shim class over jax.custom_vjp that uses ParamSpec and works with methods.

    Args:
        func: The method to decorate.
        static_argnums: The indices of the **static** arguments -- nothing to do with
            differentiation.
    """

    vjp: jax.custom_vjp[R_co]

    @override
    def __init__(self,
                 func: Callable[Concatenate[U, P], R_co],
                 *,
                 static_argnums: tuple[int, ...] = ()
                 ) -> None:
        super().__init__()
        static_argnums = tuple(sorted(static_argnums))
        self.vjp = jax.custom_vjp(func, nondiff_argnums=static_argnums)
        _ = update_wrapper(self, func, all_wrapper_assignments)

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
    def __get__(self, instance: None, owner: object = None) -> Self:
        ...

    @overload
    def __get__(self, instance: U, owner: object = None) -> Callable[P, R_co]:
        ...

    def __get__(self, instance: Any, owner: object = None) -> Callable[..., R_co]:
        if instance is None:
            return self
        # Create a partial function application corresponding to a bound method.
        return Partial(self, instance)  # type: ignore[no-untyped-call]


class custom_jvp(Generic[P, R_co]):  # noqa: N801
    """A shim class over jax.custom_jvp that uses ParamSpec.

    Args:
        func: The function to decorate.
        nondiff_argnums: The indices of the non-differentiated arguments.
    """

    @override
    def __init__(self,
                 func: Callable[P, R_co],
                 *,
                 nondiff_argnums: tuple[int, ...] = ()
                 ) -> None:
        super().__init__()
        nondiff_argnums = tuple(sorted(nondiff_argnums))
        self.jvp = jax.custom_jvp(func, nondiff_argnums=nondiff_argnums)
        _ = update_wrapper(self, func, all_wrapper_assignments)

    def defjvp(self, jvp: Callable[..., tuple[R_co, R_co]]) -> None:
        """Implement the custom forward pass of the custom derivative.

        Args:
            jvp: The custom forward pass.
        """
        _ = self.jvp.defjvp(jvp)

    def __call__(self, /, *args: P.args, **kwargs: P.kwargs) -> R_co:
        return self.jvp(*args, **kwargs)


class custom_jvp_method(Generic[U, P, R_co]):  # noqa: N801
    """A shim class over jax.custom_jvp that uses ParamSpec and works with methods.

    Args:
        func: The method to decorate.
        nondiff_argnums: The indices of the non-differentiated arguments.
    """

    @override
    def __init__(self,
                 func: Callable[Concatenate[U, P], R_co],
                 *,
                 nondiff_argnums: tuple[int, ...] = ()
                 ) -> None:
        super().__init__()
        nondiff_argnums = tuple(sorted(nondiff_argnums))
        self.jvp = jax.custom_jvp(func, nondiff_argnums=nondiff_argnums)
        _ = update_wrapper(self, func, all_wrapper_assignments)

    def defjvp(self, jvp: Callable[Concatenate[U, P], tuple[R_co, R_co]]) -> None:
        """Implement the custom forward pass of the custom derivative.

        Args:
            jvp: The custom forward pass.
        """
        _ = self.jvp.defjvp(jvp)

    def __call__(self, u: U, /, *args: P.args, **kwargs: P.kwargs) -> R_co:
        return self.jvp(u, *args, **kwargs)

    @overload
    def __get__(self, instance: None, owner: object = None) -> Self:
        ...

    @overload
    def __get__(self, instance: U, owner: object = None) -> Callable[P, R_co]:
        ...

    def __get__(self, instance: Any, owner: Any = None) -> Callable[..., R_co]:
        if instance is None:
            return self
        # Create a partial function application corresponding to a bound method.
        return Partial(self, instance)  # type: ignore[no-untyped-call]


def hessian(fun: Callable[..., U],
            argnums: int | Sequence[int] = 0,
            *,
            has_aux: bool = False,
            holomorphic: bool = False,
            reverse_only: bool = False
            ) -> Callable[..., U]:
    if not reverse_only:
        return jax.hessian(fun, argnums=argnums, has_aux=has_aux, holomorphic=holomorphic)
    return jax.jacrev(jax.jacrev(fun, argnums, has_aux=has_aux, holomorphic=holomorphic),
                      argnums, has_aux=has_aux, holomorphic=holomorphic)
