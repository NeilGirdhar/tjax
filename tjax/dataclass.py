import dataclasses
from dataclasses import MISSING, Field, FrozenInstanceError, InitVar, asdict, astuple
from dataclasses import fields as d_fields
from dataclasses import is_dataclass, replace
from functools import partial
from typing import (Any, Callable, Hashable, Iterable, List, Mapping, MutableMapping, Optional,
                    Sequence, Tuple, Type, TypeVar, overload)

from jax.tree_util import register_pytree_node

from .annotations import PyTree
from .display import display_class, display_generic, display_key_and_value
from .testing import get_relative_test_string, get_test_string, jax_allclose

__all__ = ['dataclass', 'field', 'Field', 'FrozenInstanceError', 'InitVar', 'MISSING',
           # Helper functions.
           'fields', 'asdict', 'astuple', 'replace', 'is_dataclass', 'field_names',
           'field_names_and_values', 'field_names_values_metadata', 'field_values',
           # New functions.
           'document_dataclass']


T = TypeVar('T', bound=Any)


@overload
def dataclass(*, init: bool = True, repr_: bool = True, eq: bool = True,
              order: bool = False) -> Callable[
                  [Type[T]], Type[T]]:
    ...


@overload
def dataclass(cls: Type[T], *, init: bool = True, repr_: bool = True, eq: bool = True,
              order: bool = False) -> Type[T]:
    ...


# TODO: use positional-only arguments
def dataclass(cls: Optional[Type[T]] = None, *, init: bool = True, repr_: bool = True,
              eq: bool = True, order: bool = False) -> Any:
    """
    Returns the same class as was passed in, with dunder methods added based on the fields defined
    in the class.

    Examines PEP 526 annotations to determine fields.  Default values for fields are provided using
    assignment.  To mark fields as JAX static fields rather than JAX pytree fields, use the `field`
    function.

    For example::
    ```python
    from __future__ import annotations

    from typing import ClassVar

    from tjax import dataclass, field, Array
    from jax import numpy as jnp
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

    `dataclass` includes a convenient replace method::

        w.replace(weight=3.4)

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
        return partial(dataclass, init=init, repr_=repr_, eq=eq, order=order)

    # pylint: disable=protected-access

    # Apply dataclass function to cls.
    data_clz: Type[T] = dataclasses.dataclass(cls, init=init, repr=repr_, eq=eq,
                                              order=order, frozen=True)  # type: ignore

    # Partition fields into hashed, tree, and uninitialized.
    static_fields: List[str] = []
    nonstatic_fields: List[str] = []

    for field_info in dataclasses.fields(data_clz):  # type: ignore
        if not field_info.init:
            continue
        if field_info.metadata.get('static', False):
            static_fields.append(field_info.name)
        else:
            nonstatic_fields.append(field_info.name)

    # Generate additional methods.
    def __str__(self: T) -> str:
        return str(self.display())

    def tree_flatten(x: T) -> Tuple[Sequence[PyTree], Hashable]:
        hashed = tuple(getattr(x, name) for name in static_fields)
        trees = tuple(getattr(x, name) for name in nonstatic_fields)
        return trees, hashed

    def tree_unflatten(cls: Type[T], hashed: Hashable, trees: Sequence[PyTree]) -> T:
        if not isinstance(hashed, tuple):
            raise TypeError
        hashed_args = dict(zip(static_fields, hashed))
        tree_args = dict(zip(nonstatic_fields, trees))
        return cls(**hashed_args, **tree_args)

    # Assign methods to the class.
    if data_clz.__str__ is object.__str__:
        data_clz.__str__ = __str__  # type: ignore
    if not hasattr(data_clz, 'display'):
        data_clz.display = display_dataclass  # type: ignore
    data_clz.tree_flatten = tree_flatten  # type: ignore
    data_clz.tree_unflatten = classmethod(tree_unflatten)  # type: ignore
    data_clz.replace = replace  # type: ignore

    # Assign field lists to the class.
    data_clz.nonstatic_fields = nonstatic_fields  # type: ignore
    data_clz.static_fields = static_fields  # type: ignore

    # Register the class as a JAX PyTree.
    register_pytree_node(data_clz, tree_flatten, data_clz.tree_unflatten)  # type: ignore

    # Register the dynamically-dispatched functions.
    display_generic.register(data_clz, data_clz.display)  # type: ignore
    get_test_string.register(data_clz, get_dataclass_test_string)  # type: ignore
    get_relative_test_string.register(data_clz, get_relative_dataclass_test_string)  # type: ignore
    return data_clz


def display_dataclass(value: T, show_values: bool = True, indent: int = 0) -> str:
    retval = display_class(type(value))
    for field_info in dataclasses.fields(value):  # type: ignore
        retval += display_key_and_value(
            field_info.name, getattr(value, field_info.name), "=", show_values, indent)
    return retval


def get_dataclass_test_string(actual: Any, rtol: float, atol: float) -> str:
    retval = f"{type(actual).__name__}("
    retval += ",\n".join(
        f"{fn}=" + get_test_string(getattr(actual, fn), rtol, atol)
        for fn in actual.nonstatic_fields)
    if actual.nonstatic_fields and actual.static_fields:
        retval += ',\n'
    retval += ",\n".join(
        f"{fn}=" + get_test_string(getattr(actual, fn), rtol, atol)
        for fn in actual.static_fields)
    retval += ")"
    return retval


def get_relative_dataclass_test_string(actual: Any,
                                       original_name: str,
                                       original: Any,
                                       rtol: float,
                                       atol: float) -> str:
    retval = f"{original_name}.replace("
    retval += ",\n".join(
        f"{fn}=" + get_relative_test_string(f"{original_name}.{fn}",
                                            getattr(actual, fn),
                                            getattr(original, fn),
                                            rtol,
                                            atol)
        for fn in actual.nonstatic_fields
        if not jax_allclose(getattr(actual, fn), getattr(original, fn), rtol=rtol, atol=atol))
    retval += ")"
    return retval


# NOTE: Actual return type is 'Field[T]', but we want to help type checkers
# to understand the magic that happens at runtime.
# pylint: disable=redefined-builtin
@overload  # `default` and `default_factory` are optional and mutually exclusive.
def field(*, static: bool = False, default: T, init: bool = ..., repr: bool = ...,
          hash: Optional[bool] = ..., compare: bool = ...,
          metadata: Optional[Mapping[str, Any]] = ...) -> T:
    ...


@overload
def field(*, static: bool = False, default_factory: Callable[[], T], init: bool = ...,
          repr: bool = ..., hash: Optional[bool] = ..., compare: bool = ...,
          metadata: Optional[Mapping[str, Any]] = ...) -> T:
    ...


@overload
def field(*, static: bool = False, init: bool = ..., repr: bool = ..., hash: Optional[bool] = ...,
          compare: bool = ..., metadata: Optional[Mapping[str, Any]] = ...) -> Any:
    ...


def field(*, static: bool = False, default: Any = MISSING,
          default_factory: Callable[[], Any] = MISSING, init: bool = True,  # type: ignore
          repr: bool = True, hash: Optional[bool] = None, compare: bool = True,
          metadata: Optional[Mapping[str, Any]] = None) -> Any:
    """
    Args:
        static: Indicates whether a field is a pytree or static.  Pytree fields are
            differentiated and traced.  Static fields are hashed and compared.
    """
    if metadata is None:
        metadata = {}
    return dataclasses.field(metadata={**metadata, 'static': static},
                             default=default, default_factory=default_factory, init=init, repr=repr,
                             hash=hash, compare=compare)  # type: ignore


def fields(d: Any, *, static: Optional[bool] = None) -> Iterable[Field[Any]]:
    if static is None:
        yield from d_fields(d)
    for this_field in d_fields(d):
        if this_field.metadata.get('static', False) == static:
            yield this_field


def field_names(d: Any, *, static: Optional[bool] = None) -> Iterable[str]:
    for this_field in fields(d, static=static):
        yield this_field.name


def field_names_and_values(d: Any, *, static: Optional[bool] = None) -> Iterable[Tuple[str, Any]]:
    for name in field_names(d, static=static):
        yield name, getattr(d, name)


def field_values(d: Any, *, static: Optional[bool] = None) -> Iterable[Any]:
    for name in field_names(d, static=static):
        yield getattr(d, name)


def field_names_values_metadata(d: Any, *, static: Optional[bool] = None) -> (
        Iterable[Tuple[str, Any, Mapping[str, Any]]]):
    for this_field in fields(d, static=static):
        yield this_field.name, getattr(d, this_field.name), this_field.metadata


def document_dataclass(pdoc: MutableMapping[str, Any], name: str) -> None:
    pdoc[f'{name}.static_fields'] = False
    pdoc[f'{name}.nonstatic_fields'] = False
    pdoc[f'{name}.tree_flatten'] = False
    pdoc[f'{name}.tree_unflatten'] = False
    pdoc[f'{name}.display'] = False
    pdoc[f'{name}.replace'] = False
