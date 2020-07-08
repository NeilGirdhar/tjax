from typing import Any, Hashable, List, Sequence, Tuple, Type, TypeVar

import cooperative_dataclasses as dataclasses
from jax.tree_util import register_pytree_node

from .annotations import PyTree
from .display import display_class, display_key_and_value
from .pytree_like import PyTreeLike

__all__ = ['dataclass', 'field']


T = TypeVar('T', bound=Any)


def dataclass(clz: Type[T]) -> Type[T]:
    """
    some_member: SomeType = field(pytree_like=False)
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

    # Verify that the generated class is PyTreeLike.
    assert isinstance(data_clz, PyTreeLike)

    return data_clz


def field(pytree_like: bool = True, **kwargs: Any) -> dataclasses.Field:
    return dataclasses.field(metadata={**kwargs.pop('metadata', {}),
                                       'pytree_like': pytree_like},
                             **kwargs)
