from typing import Any, Hashable, List, MutableMapping, Sequence, Tuple, Type, TypeVar

import cooperative_dataclasses as dataclasses
from cooperative_dataclasses import MISSING, Field, FrozenInstanceError, InitVar
from jax.tree_util import register_pytree_node

from .annotations import PyTree
from .display import display_class, display_key_and_value

__all__ = ['dataclass', 'field', 'Field', 'FrozenInstanceError', 'InitVar', 'MISSING']


T = TypeVar('T', bound=Any)


def dataclass(clz: Type[T]) -> Type[T]:
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

    from tjax import dataclass, field, Tensor
    from jax import numpy as jnp
    from jax import grad

    @dataclass
    class LearnedParameter:
        weight: Tensor
        constrain_positive: bool = field(pytree_like=False)
        minimum_positive_weight: ClassVar[Tensor] = 1e-6

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
    # pylint: disable=protected-access

    # Apply dataclass function to clz.
    data_clz: Type[T] = dataclasses.dataclass(frozen=True)(clz)  # type: ignore

    # Partition fields into hashed, tree, and uninitialized.
    hashed_fields: List[str] = []
    tree_fields: List[str] = []

    for field_info in dataclasses.fields(data_clz):  # type: ignore
        if not field_info.init:
            continue
        if field_info.metadata.get('pytree_like', True):
            tree_fields.append(field_info.name)
        else:
            hashed_fields.append(field_info.name)

    # Generate additional methods.
    def __repr__(self: T) -> str:
        return str(self.display())

    def display(self: T, show_values: bool = True, indent: int = 0) -> str:
        retval = display_class(type(self))
        for field_info in dataclasses.fields(data_clz):  # type: ignore
            retval += display_key_and_value(
                field_info.name, getattr(self, field_info.name), "=", show_values, indent)
        return retval

    def tree_flatten(x: T) -> Tuple[Sequence[PyTree], Hashable]:
        hashed = tuple(getattr(x, name) for name in hashed_fields)
        trees = tuple(getattr(x, name) for name in tree_fields)
        return trees, hashed

    def tree_unflatten(cls: Type[T], hashed: Hashable, trees: Sequence[PyTree]) -> T:
        if not isinstance(hashed, tuple):
            raise TypeError
        hashed_args = dict(zip(hashed_fields, hashed))
        tree_args = dict(zip(tree_fields, trees))
        return cls(**hashed_args, **tree_args)

    # Assign methods to the class.
    data_clz.__repr__ = __repr__  # type: ignore
    data_clz.display = display  # type: ignore
    data_clz.tree_flatten = tree_flatten  # type: ignore
    data_clz.tree_unflatten = classmethod(tree_unflatten)  # type: ignore

    # Assign field lists to the class.
    data_clz.tree_fields = tree_fields  # type: ignore
    data_clz.hashed_fields = hashed_fields  # type: ignore

    # Register the class as a JAX PyTree.
    register_pytree_node(data_clz, tree_flatten, data_clz.tree_unflatten)  # type: ignore

    return data_clz


def field(pytree_like: bool = True, **kwargs: Any) -> dataclasses.Field:
    """
    Args:
        pytree_like: Indicates whether a field is a pytree or static.  Pytree fields are
            differentiated and traced.
        kwargs: Any of the keyword arguments from `dataclasses.field`.
    """
    return dataclasses.field(metadata={**kwargs.pop('metadata', {}),
                                       'pytree_like': pytree_like},
                             **kwargs)


def document_dataclass(pdoc: MutableMapping[str, Any], name: str) -> None:
    pdoc[f'{name}.hashed_fields'] = False
    pdoc[f'{name}.tree_fields'] = False
    pdoc[f'{name}.tree_flatten'] = False
    pdoc[f'{name}.tree_unflatten'] = False
    pdoc[f'{name}.display'] = False
    pdoc[f'{name}.replace'] = False
