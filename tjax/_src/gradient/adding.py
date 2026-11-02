"""Transformations that add terms to updates."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, override

import optax
from optax import (
    GradientTransformation as OptaxGradientTransformation,
)

from tjax._src.annotations import (
    KeyArray,
    PyTree,
    RealNumeric,
)
from tjax.dataclasses import as_shallow_dict, dataclass, field

from .schedule import ScalarOrSchedule
from .transform import (
    OptaxWrapper,
)


@dataclass
class AddDecayedWeights[Weights: PyTree](OptaxWrapper[Weights]):
    """Add parameter scaled by `weight_decay`.

    Args:
        weight_decay: a scalar weight decay rate, or a schedule of rates.
        mask: a tree with same structure as (or a prefix of) the params PyTree,
            or a Callable that returns such a pytree given the params/updates.
            The leaves should be booleans, `True` for leaves/subtrees you want to
            apply the transformation to, and `False` for those you want to skip.
    """

    weight_decay: ScalarOrSchedule = 0.0
    mask: bool | Weights | Callable[[Weights], Any] | None = field(default=None, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.add_decayed_weights(**as_shallow_dict(self))


@dataclass
class AddNoise[Weights: PyTree](OptaxWrapper[Weights]):
    """Add gradient noise.

    References:
        [Neelakantan et al, 2014](https://arxiv.org/abs/1511.06807)

    Args:
        eta: base variance of the gaussian noise added to the gradient.
        gamma: decay exponent for annealing of the variance.
        key: PRNG key for random number generation.
    """

    eta: RealNumeric
    gamma: RealNumeric
    key: KeyArray | int | None = None

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.add_noise(**as_shallow_dict(self))
