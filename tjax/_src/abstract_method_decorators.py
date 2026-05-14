from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast, override

from .function_markers import abstract_custom_jvp_marker, abstract_jit_marker, jit_marker
from .shims import custom_jvp_method, jit


def _copy_jax_markers(
    method: object,
    *,
    jit_kwargs: dict[str, object] | None,
    abstract_jit_kwargs: dict[str, object] | None,
    custom_jvp: tuple[Callable[..., object], tuple[int, ...]] | None,
) -> object:
    if jit_kwargs is not None:
        setattr(method, jit_marker, jit_kwargs)
    if abstract_jit_kwargs is not None:
        setattr(method, abstract_jit_marker, abstract_jit_kwargs)
    if custom_jvp is not None:
        setattr(method, abstract_custom_jvp_marker, custom_jvp)
    return method


def _unwrap_marked_method(method: object) -> Callable[..., object]:
    wrapped = getattr(method, "__wrapped__", None)
    if wrapped is None:
        return cast("Callable[..., object]", method)
    return cast("Callable[..., object]", getattr(wrapped, "__wrapped__", wrapped))


def _decorate_marked_method(
    cls: type[object],
    name: str,
    original_method: Callable[..., object],
    *,
    jit_kwargs: dict[str, object] | None,
    abstract_jit_kwargs: dict[str, object] | None,
    custom_jvp: tuple[Callable[..., object], tuple[int, ...]] | None,
) -> object:
    method = original_method
    if jit_kwargs is not None:
        method = jit(original_method, **jit_kwargs)
    if custom_jvp is not None:
        jvp, nondiff_argnums = custom_jvp
        method_jvp: Any = custom_jvp_method(method, nondiff_argnums=nondiff_argnums)
        method_jvp.defjvp(jvp)
        setattr(cls, f"_original_{name}", method)
        method = method_jvp
    return _copy_jax_markers(
        method,
        jit_kwargs=jit_kwargs,
        abstract_jit_kwargs=abstract_jit_kwargs,
        custom_jvp=custom_jvp,
    )


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
        wrapped_names: set[str] = set()
        for name, original_method in list(cls.__dict__.items()):
            if not hasattr(super_cls, name):
                continue
            super_method = getattr(super_cls, name)
            if super_method is original_method:
                continue
            # Override detected.
            method = original_method
            abstract_jit_kwargs = getattr(super_method, abstract_jit_marker, None)
            jit_kwargs = abstract_jit_kwargs
            if jit_kwargs is None:
                jit_kwargs = getattr(super_method, jit_marker, None)
            custom_jvp = getattr(super_method, abstract_custom_jvp_marker, None)
            method = _decorate_marked_method(
                cls,
                name,
                original_method,
                jit_kwargs=jit_kwargs,
                abstract_jit_kwargs=abstract_jit_kwargs,
                custom_jvp=custom_jvp,
            )
            setattr(cls, name, method)
            wrapped_names.add(name)

        for base in cls.__mro__[1:]:
            for name, inherited_method in base.__dict__.items():
                if (
                    name in cls.__dict__
                    or name in wrapped_names
                    or name.startswith("_")
                    or getattr(inherited_method, "__isabstractmethod__", False)
                ):
                    continue
                abstract_jit_kwargs = getattr(inherited_method, abstract_jit_marker, None)
                jit_kwargs = abstract_jit_kwargs
                if jit_kwargs is None:
                    jit_kwargs = getattr(inherited_method, jit_marker, None)
                custom_jvp = getattr(inherited_method, abstract_custom_jvp_marker, None)
                if jit_kwargs is None and custom_jvp is None:
                    continue
                method = _decorate_marked_method(
                    cls,
                    name,
                    _unwrap_marked_method(inherited_method),
                    jit_kwargs=jit_kwargs,
                    abstract_jit_kwargs=abstract_jit_kwargs,
                    custom_jvp=custom_jvp,
                )
                setattr(cls, name, method)
                wrapped_names.add(name)


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
        fun._abstract_custom_jvp = (  # type: ignore   # noqa: SLF001
            jvp,
            nondiff_argnums,
        )
        return fun

    return decorator  # type: ignore
