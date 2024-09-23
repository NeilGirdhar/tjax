from __future__ import annotations

from dataclasses import KW_ONLY, InitVar, dataclass
from typing import Any

from flax import nnx
from typing_extensions import dataclass_transform, override

from .helpers import field


def module_field(*, init: bool = False) -> Any:
    """A field that contains submodules."""
    return field(init=init, default=None, kw_only=True)


@dataclass_transform(field_specifiers=(module_field, field))
class _DataClassModule(nnx.Module):
    @override
    def __init_subclass__(cls,
                          *,
                          init: bool = True,
                          repr: bool = True,
                          eq: bool = True,
                          order: bool = False,
                          kw_only: bool = False,
                          **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs, experimental_pytree=True)
        _ = dataclass(init=init, repr=repr, eq=eq, order=order, kw_only=kw_only)(cls)


class DataClassModule(_DataClassModule):
    _: KW_ONLY
    rngs: InitVar[nnx.Rngs] = field()

    def __post_init__(self, rngs: nnx.Rngs) -> None:
        pass
