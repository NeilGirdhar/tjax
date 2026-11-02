"""Transformations that constrain updates or parameters."""

from __future__ import annotations

from typing import override

import optax
from optax import (
    GradientTransformation as OptaxGradientTransformation,
)

from tjax._src.annotations import (
    PyTree,
)
from tjax.dataclasses import as_shallow_dict, dataclass

from .transform import (
    OptaxWrapper,
)


@dataclass
class KeepParamsNonnegative[Weights: PyTree](OptaxWrapper[Weights]):
    """Modifies the updates to keep parameters non-negative, i.e. >= 0."""

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.keep_params_nonnegative(**as_shallow_dict(self))


@dataclass
class ZeroNans[Weights: PyTree](OptaxWrapper[Weights]):
    """A transformation which replaces NaNs with 0."""

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.zero_nans(**as_shallow_dict(self))
