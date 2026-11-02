"""Transformations that record measurements of the updates."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, override

import optax
from optax import (
    GradientTransformation as OptaxGradientTransformation,
)

from tjax._src.annotations import (
    PyTree,
    RealNumeric,
)
from tjax.dataclasses import dataclass, field

from .transform import (
    GradientTransformation,
    OptaxWrapper,
    to_optax,
)


@dataclass
class Snapshot[Weights: PyTree](OptaxWrapper[Weights]):
    """Store a measurement of the updates in the state, and pass the updates through.

    Args:
        measure_name: The key under which the state stores the measurement.
        measure: A function from the updates to the measurement.
    """

    measure_name: str = field(static=True)
    measure: Callable[[PyTree], PyTree] = field(static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.snapshot(self.measure_name, self.measure)


@dataclass
class MeasureWithEma[Weights: PyTree](OptaxWrapper[Weights]):
    """Store an exponential moving average of a measurement of the updates in the state.

    The updates pass through unchanged.

    Args:
        measure: A function from the updates to the measurement.
        decay: The decay rate of the moving average.
        debias: Whether to debias the moving average.
        accumulator_dtype: Optional dtype for the moving average.
    """

    measure: Callable[[PyTree], PyTree] = field(static=True)
    decay: RealNumeric
    debias: bool = field(default=True, static=True)
    accumulator_dtype: Any = field(default=None, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.measure_with_ema(
            self.measure, self.decay, debias=self.debias, accumulator_dtype=self.accumulator_dtype
        )


@dataclass
class Monitor[Weights: PyTree](OptaxWrapper[Weights]):
    """Monitor stateful measurements of the updates, and pass the updates through.

    The state stores the latest value of each measurement.

    Args:
        measures: A map from names to transformations whose output is the measurement.  Each one
            sees the updates and parameters, and its state is kept between steps.
        functions: A map from names to stateless functions of the updates.
    """

    measures: Mapping[str, GradientTransformation[Any, Weights]] = field(default_factory=dict)
    functions: Mapping[str, Callable[[PyTree], PyTree]] = field(default_factory=dict, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.monitor(
            {name: to_optax(measure) for name, measure in self.measures.items()}
            | dict(self.functions)
        )
