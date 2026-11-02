"""Transformations that clip updates."""

from __future__ import annotations

from typing import override

import optax
from optax import (
    GradientTransformation as OptaxGradientTransformation,
)

from tjax._src.annotations import (
    PyTree,
    RealNumeric,
)
from tjax.dataclasses import as_shallow_dict, dataclass, field

from .transform import (
    OptaxWrapper,
)


@dataclass
class Clip[Weights: PyTree](OptaxWrapper[Weights]):
    """Clips updates element-wise, to be in ``[-max_delta, +max_delta]``.

    Args:
        max_delta: The maximum absolute value for each element in the update.
    """

    max_delta: RealNumeric

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.clip(**as_shallow_dict(self))


@dataclass
class ClipByGlobalNorm[Weights: PyTree](OptaxWrapper[Weights]):
    """Clips updates using their global norm.

    References:
        Pascanu et al., `On the difficulty of training Recurrent Neural Networks
            <https://arxiv.org/abs/1211.5063>`_, 2012

    Args:
        max_norm: The maximum global norm for an update.
    """

    max_norm: RealNumeric

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.clip_by_global_norm(**as_shallow_dict(self))


@dataclass
class ClipByBlockRMS[Weights: PyTree](OptaxWrapper[Weights]):
    """Clips updates to a max rms for the gradient of each param vector or matrix.

    Args:
        threshold: The maximum rms for the gradient of each param vector or matrix.
    """

    threshold: RealNumeric

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.clip_by_block_rms(**as_shallow_dict(self))


@dataclass
class AdaptiveGradClip[Weights: PyTree](OptaxWrapper[Weights]):
    """Clips updates to be at most ``clipping * parameter_norm``, unit-wise.

    References:
        Brock et al., `High-Performance Large-Scale Image Recognition Without Normalization
            <https://arxiv.org/abs/2102.06171>`_, 2021

    Args:
        clipping: The maximum allowed ratio of update norm to parameter norm.
        eps: An epsilon term to prevent clipping of zero-initialized params.
        axis: Axis or axes along which to compute the unit-wise norm. If None, uses default behavior
            based on input dimensions (including Conv3D, ndim=5). Provide axis for custom parameter
            shapes beyond the defaults.
    """

    clipping: RealNumeric
    eps: RealNumeric = 0.001
    axis: int | tuple[int, ...] | None = field(default=None, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.adaptive_grad_clip(**as_shallow_dict(self))
