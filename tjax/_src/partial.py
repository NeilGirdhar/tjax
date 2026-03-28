from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from reprlib import recursive_repr
from typing import Any, cast, override

from jax.tree_util import register_pytree_node_class

from .annotations import PyTree


@register_pytree_node_class
class Partial[R]:
    """A version of functools.partial that returns a pytree.

    Use it for partial function evaluation in a way that is compatible with JAX's transformations,
    e.g., ``Partial(func, *args, **kwargs)``.

    JAX's version of this class in jax.tree_util assumes that all parameters are dynamic and the
    callable is static.  This version allows you to specify which parameters are static and whether
    the callable is static.

    Args:
        function: The function being applied.
        args: The static and dynamic applied positional arguments.
        callable_is_static: Whether the function callable is static.
        static_argnums: The indices of the applied positional arguments that are static.
        static_kwargs: The static applied keyword arguments.
        dynamic_kwargs: The dynamic applied keyword arguments.
    """

    @override
    def __init__(
        self,
        function: Partial[R] | Callable[..., R],
        /,
        *args: object,
        callable_is_static: bool = True,
        static_argnums: tuple[int, ...] = (),
        static_kwargs: Mapping[str, Any] | None = None,
        **dynamic_kwargs: object,
    ) -> None:
        super().__init__()
        if isinstance(function, Partial) and callable_is_static:
            raise TypeError
            # Could collapse the arguments here, but the benefit is small.
        self.function = function
        self.callable_is_static = callable_is_static
        self.args = args
        self.static_argnums = static_argnums
        self.static_kwargs = {} if static_kwargs is None else static_kwargs
        self.dynamic_kwargs = dynamic_kwargs

    def tree_flatten(self: Partial[R]) -> tuple[Sequence[PyTree], Hashable]:
        argument_slots = (
            self.callable_is_static,
            *(i in self.static_argnums for i in range(len(self.args))),
        )
        static_args: list[Any] = []
        dynamic_args: list[Any] = []
        for value, is_static in zip((self.function, *self.args), argument_slots, strict=True):
            if is_static:
                static_args.append(value)
            else:
                dynamic_args.append(value)
        static_kwargs = tuple((key, self.static_kwargs[key]) for key in sorted(self.static_kwargs))
        return (
            (tuple(dynamic_args), self.dynamic_kwargs),
            (argument_slots, tuple(static_args), static_kwargs),
        )

    @classmethod
    def tree_unflatten(cls, static: Hashable, trees: Sequence[PyTree]) -> Partial[R]:
        if not isinstance(static, tuple):
            raise RuntimeError  # noqa: TRY004

        argument_slots, static_args, static_kwarg_items = static
        assert isinstance(argument_slots, tuple)
        assert isinstance(static_kwarg_items, tuple)
        static_kwargs = dict(static_kwarg_items)

        if not isinstance(static_args, tuple):
            raise RuntimeError  # noqa: TRY004

        dynamic_args, dynamic_kwargs = trees

        if not isinstance(dynamic_args, tuple):
            raise RuntimeError  # noqa: TRY004
        if not isinstance(dynamic_kwargs, dict):
            raise RuntimeError  # noqa: TRY004

        dynamic_kwargs = cast("dict[str, Any]", dynamic_kwargs)
        args = cls._unpartition_args(argument_slots, static_args, dynamic_args)
        callable_is_static = argument_slots[0]
        static_argnums = tuple(
            i - 1 for i, is_static in enumerate(argument_slots[1:], start=1) if is_static
        )

        return Partial[R](
            *args,
            callable_is_static=callable_is_static,
            static_argnums=static_argnums,
            static_kwargs=static_kwargs,
            **dynamic_kwargs,
        )

    # Magic methods --------------------------------------------------------------------------------
    def __call__(self, *args: object, **kwargs: object) -> R:
        keywords = {**self.static_kwargs, **self.dynamic_kwargs, **kwargs}
        return self.function(*self.args, *args, **keywords)

    @recursive_repr()
    def __repr__(self) -> str:
        qualname = type(self).__qualname__
        args = [repr(self.function)]
        args.extend(repr(x) for x in self.args)
        args.extend(
            [
                f"callable_is_static={self.callable_is_static}",
                f"static_argnums={self.static_argnums}",
                f"static_kwargs={self.static_kwargs}",
            ]
        )
        args.extend(f"{k}={v!r}" for (k, v) in self.dynamic_kwargs.items())
        return f"{qualname}({', '.join(args)})"

    @classmethod
    def _unpartition_args(
        cls,
        argument_slots: tuple[bool, ...],
        static_args: tuple[Any, ...],
        dynamic_args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        static_iter = iter(static_args)
        dynamic_iter = iter(dynamic_args)
        args = [
            next(static_iter) if is_static else next(dynamic_iter) for is_static in argument_slots
        ]
        return tuple(args)
