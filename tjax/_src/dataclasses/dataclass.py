from __future__ import annotations

import dataclasses
from typing import (Any, Callable, ClassVar, Dict, Hashable, List, Optional, Protocol, Sequence,
                    Tuple, Type, Union, cast, overload, runtime_checkable)

from jax.tree_util import AttributeKeyPathEntry, register_keypaths, register_pytree_node
from typing_extensions import dataclass_transform

from ..annotations import PyTree
from ..testing import get_relative_test_string, get_test_string, tree_allclose
from .helpers import field

__all__ = ['dataclass', 'DataclassInstance', 'TDataclassInstance']


@runtime_checkable
class DataclassInstance(Protocol):
    __dataclass_fields__: ClassVar[Dict[str, dataclasses.Field[Any]]]


@runtime_checkable
class TDataclassInstance(DataclassInstance, Protocol):
    static_fields: ClassVar[List[str]]
    dynamic_fields: ClassVar[List[str]]


@overload
@dataclass_transform(frozen_default=True, field_specifiers=(field,))
def dataclass(*, init: bool = True, repr_: bool = True, eq: bool = True,
              order: bool = False) -> Callable[[Type[Any]], Type[TDataclassInstance]]:
    ...


@overload
@dataclass_transform(frozen_default=True, field_specifiers=(field,))
def dataclass(cls: Type[Any], /, *, init: bool = True, repr_: bool = True, eq: bool = True,
              order: bool = False) -> Type[TDataclassInstance]:
    ...


@dataclass_transform(frozen_default=True, field_specifiers=(field,))
def dataclass(cls: Optional[Type[Any]] = None, /, *, init: bool = True, repr_: bool = True,
              eq: bool = True, order: bool = False
              ) -> Union[Type[TDataclassInstance], Callable[[Type[Any]], Type[TDataclassInstance]]]:
    """
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
    from tjax import Array, dataclass
    from tjax.dataclasses import field
    from jax import grad

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

    Another benefit is the display of dataclasses.  `print(new_w)` gives::
    ```
    LearnedParameter
        weight=Jax Array ()
                2.0003
        constrain_positive=True
    ```
    """
    if cls is None:
        def f(x: Type[Any], /) -> Type[TDataclassInstance]:
            return dataclass(x, init=init, repr_=repr_, eq=eq, order=order)
        return f  # Type checking support partial is poor.
    non_none_cls = cls

    # Apply dataclass function to cls.
    data_clz: Type[TDataclassInstance] = cast(Type[TDataclassInstance],
                                              dataclasses.dataclass(init=init, repr=repr_, eq=eq,
                                                                    order=order, frozen=True)(cls))

    # Partition fields into hashed, tree, and uninitialized.
    static_fields: List[str] = []
    dynamic_fields: List[str] = []

    for field_info in dataclasses.fields(data_clz):
        if not field_info.init:
            continue
        if field_info.metadata.get('static', False):
            static_fields.append(field_info.name)
        else:
            dynamic_fields.append(field_info.name)

    # Generate additional methods.
    def tree_flatten(x: Any) -> Tuple[Sequence[PyTree], Hashable]:
        hashed = tuple(getattr(x, name) for name in static_fields)
        trees = tuple(getattr(x, name) for name in dynamic_fields)
        return trees, hashed

    def tree_unflatten(hashed: Hashable, trees: Sequence[PyTree]) -> Any:
        if not isinstance(hashed, tuple):
            raise TypeError
        hashed_args = dict(zip(static_fields, hashed))
        tree_args = dict(zip(dynamic_fields, trees))
        return non_none_cls(**hashed_args, **tree_args)

    def keypaths(_: Any) -> List[AttributeKeyPathEntry]:
        return [AttributeKeyPathEntry(name) for name in dynamic_fields]

    # Assign field lists to the class.
    data_clz.dynamic_fields = dynamic_fields
    data_clz.static_fields = static_fields

    # Register the class as a JAX PyTree.
    register_pytree_node(data_clz, tree_flatten, tree_unflatten)
    register_keypaths(data_clz, keypaths)

    # Register the dynamically-dispatched functions.
    get_test_string.register(data_clz, get_dataclass_test_string)
    get_relative_test_string.register(data_clz, get_relative_dataclass_test_string)
    return data_clz


def get_dataclass_test_string(actual: Any, rtol: float, atol: float) -> str:
    retval = f"{type(actual).__name__}("
    retval += ",\n".join(((f"{fn.name}=" if fn.kw_only else "")
                          + get_test_string(getattr(actual, fn.name), rtol, atol))
                         for fn in dataclasses.fields(actual))
    retval += ")"
    return retval


def get_relative_dataclass_test_string(actual: Any,
                                       original_name: str,
                                       original: Any,
                                       rtol: float,
                                       atol: float) -> str:
    retval = f"replace({original_name}, "
    retval += ",\n".join(
        f"{fn}=" + get_relative_test_string(f"{original_name}.{fn}",
                                            getattr(actual, fn),
                                            getattr(original, fn),
                                            rtol,
                                            atol)
        for fn in actual.dynamic_fields
        if not tree_allclose(getattr(actual, fn), getattr(original, fn), rtol=rtol, atol=atol))
    retval += ")"
    return retval
