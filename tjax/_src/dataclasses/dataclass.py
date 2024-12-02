from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, ClassVar, Protocol, cast, overload, runtime_checkable

from jax.tree_util import register_dataclass
from typing_extensions import dataclass_transform

from ..testing import get_relative_test_string, get_test_string, tree_allclose
from .helpers import field


@runtime_checkable
class DataclassInstance(Protocol):
    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[Any]]]


@runtime_checkable
class TDataclassInstance(DataclassInstance, Protocol):
    static_fields: ClassVar[list[str]]
    dynamic_fields: ClassVar[list[str]]


@overload
@dataclass_transform(frozen_default=True, field_specifiers=(field,))
def dataclass(*, init: bool = True, repr: bool = True, eq: bool = True,
              order: bool = False, frozen: bool = True
              ) -> Callable[[type[Any]], type[TDataclassInstance]]: ...
@overload
@dataclass_transform(frozen_default=True, field_specifiers=(field,))
def dataclass(cls: type[Any], /, *, init: bool = True, repr: bool = True,
              eq: bool = True, order: bool = False, frozen: bool = True
              ) -> type[TDataclassInstance]: ...
@dataclass_transform(frozen_default=True, field_specifiers=(field,))
def dataclass(cls: type[Any] | None = None, /, *, init: bool = True, repr: bool = True,  # noqa: A002
              eq: bool = True, order: bool = False, frozen: bool = True
              ) -> type[TDataclassInstance] | Callable[[type[Any]], type[TDataclassInstance]]:
    """A dataclass creator that creates a Jax pytree.

    Returns the same class as was passed in, with dunder methods added based on the fields defined
    in the class.

    Examines PEP 526 annotations to determine fields.  Default values for fields are provided using
    assignment.

    To mark fields as static fields rather than JAX pytree fields, use the `field` function.
    In JAX, a static attribute is one that induces recompilation of a function when it changes, and
    consequently there is more flexibility about what can be done with such an attribute.

    For example::
    ```python
    from __future__ import annotations

    from typing import ClassVar

    import jax.numpy as jnp
    from tjax.dataclasses import dataclass, field
    from jax import Array, grad

    @dataclass
    class LearnedParameter:
        weight: Array
        constrain_positive: bool = field(static=True)
        minimum_positive_weight: ClassVar[Array] = 1e-6

        def trained(self,
                    self_bar: LearnedParameter,
                    learning_rate: float) -> LearnedParameter:
            weight_bar = self_bar.weight
            weight = self.weight - weight_bar * learning_rate
            if self.constrain_positive:
                weight = jnp.maximum(weight, self.minimum_positive_weight)
            return LearnedParameter(weight=weight,
                                    constrain_positive=self.constrain_positive)

    def loss(w: LearnedParameter) -> float:
        return jnp.square(w.weight - 3.3)

    w = LearnedParameter(2.0, True)
    w_bar = grad(loss)(w)
    new_w = w.trained(w_bar, 1e-4)
    ```

    Since this dataclass is a pytree, all of JAX's functions that accept pytrees work with it,
    including iteration, differentiation, and `jax.tree_util` functions.
    """
    if cls is None:
        def f(x: type[Any], /) -> type[TDataclassInstance]:
            return dataclass(x, init=init, repr=repr, eq=eq, order=order, frozen=frozen)
        return f  # Type checking support partial is poor.

    # Apply dataclass function to cls.
    data_clz: type[TDataclassInstance] = cast(
            'type[TDataclassInstance]',
            dataclasses.dataclass(init=init, repr=repr, eq=eq, order=order, frozen=frozen)(cls))

    # Partition fields into static, and dynamic; and assign these to the class.
    static_fields: list[str] = []
    dynamic_fields: list[str] = []
    for field_info in dataclasses.fields(data_clz):
        if not field_info.init:
            continue
        if field_info.metadata.get('static', False):
            static_fields.append(field_info.name)
        else:
            dynamic_fields.append(field_info.name)
    data_clz.static_fields = static_fields
    data_clz.dynamic_fields = dynamic_fields
    _ = register_dataclass(data_clz, dynamic_fields, static_fields)

    # Register the dynamically-dispatched functions.
    _ = get_test_string.register(data_clz, get_dataclass_test_string)
    _ = get_relative_test_string.register(data_clz, get_relative_dataclass_test_string)
    return data_clz


def get_dataclass_test_string(actual: Any, rtol: float, atol: float) -> str:
    retval = f"{type(actual).__name__}("
    retval += ",\n".join(((f"{field_object.name}=" if field_object.kw_only else "")
                          + get_test_string(getattr(actual, field_object.name), rtol, atol))
                         for field_object in dataclasses.fields(actual)
                         if field_object.repr)
    retval += ")"
    return retval


def get_relative_dataclass_test_string(actual: Any,
                                       original_name: str,
                                       original: Any,
                                       rtol: float,
                                       atol: float) -> str:
    repr_dict = {field.name: field.repr
                 for field in dataclasses.fields(actual)}
    retval = f"replace({original_name}, "
    retval += ",\n".join(
        f"{field_name}=" + get_relative_test_string(f"{original_name}.{field_name}",
                                                    getattr(actual, field_name),
                                                    getattr(original, field_name),
                                                    rtol,
                                                    atol)
        for field_name in actual.dynamic_fields
        if repr_dict[field_name]
        if not tree_allclose(getattr(actual, field_name), getattr(original, field_name),
                             rtol=rtol, atol=atol))
    retval += ")"
    return retval
