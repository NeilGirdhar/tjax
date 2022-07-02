from __future__ import annotations

from reprlib import recursive_repr
from typing import (Any, Callable, Dict, Generic, Hashable, Mapping, Optional, Sequence, Tuple,
                    TypeVar, Union, cast)

from jax.tree_util import register_pytree_node_class

from .annotations import PyTree

__all__ = ['Partial']


R = TypeVar('R')


@register_pytree_node_class
class Partial(Generic[R]):
    """
    A version of functools.partial that returns a pytree.

    Use it for partial function evaluation in a way that is compatible with JAX's transformations,
    e.g., ``Partial(func, *args, **kwargs)``.

    JAX's version of this class in jax.tree_util assumes that all parameters are dynamic and the
    callable is static.  This version allows you to specify which parameters are static and whether
    the callable is static.
    """
    def __init__(self,
                 function: Union[Partial[R], Callable[..., R]],
                 /,
                 *args: Any,
                 callable_is_static: bool = True,
                 static_argnums: Tuple[int, ...] = (),
                 static_kwargs: Optional[Mapping[str, Any]] = None,
                 **dynamic_kwargs: Any) -> None:
        """
        Args:
            function: The function being applied.
            args: The static and dynamic applied positional arguments.
            callable_is_static: Whether the function callable is static.
            static_argnums: The indices of the applied positional arguments that are static.
            static_kwargs: The static applied keyword arguments.
            dynamic_kwargs: The dynamic applied keyword arguments.
        """
        super().__init__()
        if isinstance(function, Partial):
            if callable_is_static:
                raise TypeError
            # Could collapse the arguments here, but the benefit is small.
        self.function = function
        self.callable_is_static = callable_is_static
        self.args = args
        self.static_argnums = static_argnums
        self.static_kwargs = {} if static_kwargs is None else static_kwargs
        self.dynamic_kwargs = dynamic_kwargs

    def tree_flatten(self: Partial[R]) -> Tuple[Sequence[PyTree], Hashable]:
        static_args, dynamic_args = self._partition_args()
        static_kwargs = tuple((key, self.static_kwargs[key]) for key in sorted(self.static_kwargs))
        return ((dynamic_args, self.dynamic_kwargs),
                (self.callable_is_static, self.static_argnums, static_args, static_kwargs))

    @classmethod
    def tree_unflatten(cls,
                       static: Hashable,
                       trees: Sequence[PyTree]) -> Partial[R]:
        if not isinstance(static, tuple):
            raise RuntimeError

        callable_is_static, static_argnums, static_args, static_kwarg_items = static
        static_kwargs = dict(static_kwarg_items)

        if not isinstance(static_args, tuple):
            raise RuntimeError

        dynamic_args, dynamic_kwargs = trees

        if not isinstance(dynamic_args, tuple):
            raise RuntimeError
        if not isinstance(dynamic_kwargs, dict):
            raise RuntimeError

        dynamic_kwargs = cast(Dict[str, Any], dynamic_kwargs)
        args = cls._unpartition_args(callable_is_static, static_argnums, static_args, dynamic_args)

        return Partial[R](*args,
                          callable_is_static=callable_is_static,
                          static_argnums=static_argnums,
                          static_kwargs=static_kwargs,
                          **dynamic_kwargs)

    # Magic methods --------------------------------------------------------------------------------
    def __call__(self, *args: Any, **kwargs: Any) -> R:
        keywords = {**self.static_kwargs, **self.dynamic_kwargs, **kwargs}
        return self.function(*self.args, *args, **keywords)

    @recursive_repr()
    def __repr__(self) -> str:
        qualname = type(self).__qualname__
        args = [repr(self.function)]
        args.extend(repr(x) for x in self.args)
        args.append(f"callable_is_static={self.callable_is_static}")
        args.append(f"static_argnums={self.static_argnums}")
        args.append(f"static_kwargs={self.static_kwargs}")
        args.extend(f"{k}={v!r}" for (k, v) in self.dynamic_kwargs.items())
        return f"{qualname}({', '.join(args)})"

    # Private methods ------------------------------------------------------------------------------
    def _partition_args(self) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
        # Partition self.args into static and dynamic arguments.
        static_args = []
        dynamic_args = []

        static_argnums = set(self.static_argnums)

        def _append(is_static: bool, value: Any) -> None:
            if is_static:
                static_args.append(value)
            else:
                dynamic_args.append(value)

        _append(self.callable_is_static, self.function)
        for i, value in enumerate(self.args):
            _append(i in static_argnums, value)

        return tuple(reversed(static_args)), tuple(reversed(dynamic_args))

    @classmethod
    def _unpartition_args(cls,
                          callable_is_static: bool,
                          static_argnums: Tuple[int, ...],
                          static_args: Tuple[Any, ...],
                          dynamic_args: Tuple[Any, ...]) -> Tuple[Any, ...]:
        static_arg_list = list(static_args)
        dynamic_arg_list = list(dynamic_args)
        args = []
        for i in range(len(static_args) + len(dynamic_args)):
            is_static = (callable_is_static
                         if i == 0
                         else i - 1 in static_argnums)
            if is_static:
                args.append(static_arg_list.pop())
            else:
                args.append(dynamic_arg_list.pop())
        return tuple(args)
