from __future__ import annotations

from functools import partial
from typing import (Any, Callable, Dict, Generic, Hashable, Iterable, Mapping, Sequence, Tuple,
                    Type, TypeVar, cast)

from jax.tree_util import register_pytree_node_class

from .annotations import PyTree

__all__ = ['Partial']


R = TypeVar('R')


@register_pytree_node_class
class Partial(partial, Generic[R]):
    """
    A version of functools.partial that returns a pytree.

    Use it for partial function evaluation in a way that is compatible with JAX's transformations,
    e.g., ``Partial(func, *args, **kwargs)``.
    """

    callable_is_static: bool
    static_argnums: Tuple[int, ...]
    static_kwargs: Mapping[str, Any]

    # TODO: use positional-only arguments.
    def __new__(cls: Type[T],
                func: Callable[..., R],
                # /,
                *args: Any,
                callable_is_static: bool = True,
                static_argnums: Tuple[int, ...] = (),
                static_kwargs: Mapping[str, Any] = {},
                **kwargs: Any) -> T:
        """
        Args:
            func: The function being applied.
            args: The applied positional arguments.
            callable_is_static: Whether the function callable is static.
            static_argnums: The indices of the applied positional arguments that are static.
            static_kwargs: The key-value pairs representing applied keyword arguments that are
                static.
            kwargs: The applied keyword arguments.
        """
        if callable_is_static and isinstance(func, Partial):
            raise TypeError
        retval = super().__new__(cls, func, *args, **kwargs)  # type: ignore
        retval.callable_is_static = callable_is_static
        retval.static_argnums = set(static_argnums)
        retval.static_kwargs = static_kwargs
        return retval

    def tree_flatten(self: Partial[R]) -> Tuple[Sequence[PyTree], Hashable]:
        static_args = []
        tree_args = []

        def _append(is_static: bool, value: Any) -> None:
            if is_static:
                static_args.append(value)
            else:
                tree_args.append(value)

        _append(self.callable_is_static, self.func)
        for i, value in enumerate(self.args):
            _append(i in self.static_argnums, value)

        return ((list(reversed(tree_args)), self.keywords),
                (self.callable_is_static, self.static_argnums,
                 list(reversed(static_args)), self.static_kwargs))

    @classmethod
    def tree_unflatten(cls: Type[R],
                       static: Hashable,
                       trees: Sequence[PyTree]) -> Partial[R]:
        if not isinstance(static, Iterable):
            raise RuntimeError

        callable_is_static, static_argnums, static_args, static_kwargs = static

        if not isinstance(static_args, list):
            raise RuntimeError

        tree_args, tree_kwargs = trees

        if not isinstance(tree_args, list):
            raise RuntimeError
        if not isinstance(tree_kwargs, dict):
            raise RuntimeError

        tree_kwargs = cast(Dict[str, Any], tree_kwargs)

        args = []
        for i in range(len(static_args) + len(tree_args)):
            if i == 0:
                is_static = callable_is_static
            else:
                is_static = i - 1 in static_argnums
            if is_static:
                args.append(static_args.pop())
            else:
                args.append(tree_args.pop())

        return Partial[R](*args,
                          callable_is_static=callable_is_static,
                          static_argnums=static_argnums,
                          static_kwargs=static_kwargs,
                          **tree_kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> R:
        return super().__call__(*args, **self.static_kwargs, **kwargs)


T = TypeVar('T', bound=Partial)
