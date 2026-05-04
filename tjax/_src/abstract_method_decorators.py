from __future__ import annotations

from collections.abc import Callable
from typing import Any, override

from .function_markers import abstract_custom_jvp_marker, abstract_jit_marker
from .shims import custom_jvp_method, jit


class JaxAbstractClass:
    """Base class that automatically applies JAX decorators to method overrides.

    When a subclass overrides a method that was decorated with
    :func:`abstract_jit` or :func:`abstract_custom_jvp`, the override is
    automatically wrapped with the corresponding JAX decorator
    (:func:`~tjax._src.shims.jit` or
    :func:`~tjax._src.shims.custom_jvp_method`) at class creation time.

    This removes boilerplate from concrete subclasses: they only need to
    provide the implementation, not re-apply the decorator.
    """

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
                setattr(cls, f"_original_{name}", method)
                method = method_jvp
            setattr(cls, name, method)


def abstract_jit[F: Callable[..., Any]](fun: F, **kwargs: object) -> F:
    """Mark an abstract method so that its overrides are automatically JIT-compiled.

    Any keyword arguments are forwarded to :func:`~tjax._src.shims.jit` when
    the override is registered by :class:`JaxAbstractClass`.
    """
    setattr(fun, abstract_jit_marker, kwargs)
    return fun


def abstract_custom_jvp[**P, R_co](
    jvp: Callable[..., tuple[R_co, R_co]], nondiff_argnums: tuple[int, ...] = ()
) -> Callable[[Callable[P, R_co]], Callable[P, R_co]]:
    """Mark an abstract method so that its overrides get a custom JVP rule.

    Args:
        jvp: The custom JVP function, called with ``(primals, tangents)`` and
            returning ``(primal_out, tangent_out)``.
        nondiff_argnums: Indices of arguments that should not be differentiated.

    The decorated method will have :func:`~tjax._src.shims.custom_jvp_method`
    applied automatically by :class:`JaxAbstractClass` when a subclass
    provides an override.
    """

    def decorator(fun: Callable[P, R_co]) -> Callable[P, R_co]:
        fun._abstract_custom_jvp = (  # ty: ignore   # noqa: SLF001
            jvp,
            nondiff_argnums,
        )
        return fun

    return decorator  # ty: ignore
