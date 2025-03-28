from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from typing_extensions import ParamSpec, override

from .function_markers import abstract_custom_jvp_marker, abstract_jit_marker
from .shims import custom_jvp_method, jit

R_co = TypeVar('R_co', covariant=True)
F = TypeVar('F', bound=Callable[..., Any])
P = ParamSpec('P')


class JaxAbstractClass:
    """A class with abstract methods whose overrides need to be decorated with Jax decorators."""
    @override
    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        super_cls = super(cls, cls)
        for name, original_method in list(cls.__dict__.items()):
            if not hasattr(super_cls, name):
                continue
            super_method = getattr(super_cls, name)
            if super_method is original_method:
                continue
            # Override detected.
            method = original_method
            if (kwargs := getattr(super_method, abstract_jit_marker, None)) is not None:
                method = jit(original_method, **kwargs)
            if hasattr(super_method, abstract_custom_jvp_marker):
                jvp, nondiff_argnums = super_method._abstract_custom_jvp  # noqa: SLF001
                method_jvp: Any = custom_jvp_method(method, nondiff_argnums=nondiff_argnums)
                method_jvp.defjvp(jvp)
                setattr(cls, f'_original_{name}', method)
                method = method_jvp
            setattr(cls, name, method)


def abstract_jit(fun: F, **kwargs: object) -> F:
    """An abstract method whose override need to be jitted."""
    setattr(fun, abstract_jit_marker, kwargs)
    return fun


def abstract_custom_jvp(jvp: Callable[..., tuple[R_co, R_co]],
                        nondiff_argnums: tuple[int, ...] = ()
    ) -> Callable[[Callable[P, R_co]], Callable[P, R_co]]:
    """An abstract method whose override need to be decorated with custom_jvp_method."""
    def decorator(fun: Callable[P, R_co]) -> Callable[P, R_co]:
        fun._abstract_custom_jvp = (jvp,  # type: ignore # noqa: SLF001 # pyright: ignore
                                    nondiff_argnums)
        return fun
    return decorator
