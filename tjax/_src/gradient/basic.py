"""Elementary transformations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, override

import optax
from optax import (
    GradientTransformation as OptaxGradientTransformation,
)

from tjax._src.annotations import (
    PyTree,
)
from tjax.dataclasses import as_shallow_dict, dataclass, field

from .transform import (
    OptaxWrapper,
)


@dataclass
class Identity[Weights: PyTree](OptaxWrapper[Weights]):
    """Stateless identity transformation that leaves input gradients untouched."""

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.identity(**as_shallow_dict(self))


@dataclass
class SetToZero[Weights: PyTree](OptaxWrapper[Weights]):
    """Stateless transformation that maps input gradients to zero."""

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.set_to_zero(**as_shallow_dict(self))


@dataclass
class Stateless[Weights: PyTree](OptaxWrapper[Weights]):
    """Create a stateless transformation from a function of the updates and parameters.

    Args:
        f: A function from the updates and parameters (or ``None``) to new updates.
    """

    f: Callable[[PyTree, PyTree], PyTree] = field(static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.stateless(self.f)


@dataclass
class StatelessWithTreeMap[Weights: PyTree](OptaxWrapper[Weights]):
    """Create a stateless transformation that maps a function over the update leaves.

    Args:
        f: A function from an update leaf and its parameter leaf (or ``None``) to a new update
            leaf.
    """

    f: Callable[[Any, Any], Any] = field(static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.stateless_with_tree_map(self.f)
