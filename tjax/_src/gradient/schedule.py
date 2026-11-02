"""Step-size schedules that are pytrees."""

from __future__ import annotations

from collections.abc import Callable

from jax.typing import ArrayLike

from tjax._src.annotations import IntegralNumeric, JaxArray
from tjax.dataclasses import dataclass, field


@dataclass
class Schedule:
    """The schedule of step sizes.

    This class differs from optax.Schedule in that it's a pytree.  It can therefore be passed
    dynamically (since it marks the callable as static).  This allows it to participate in the union
    ScalarOrSchedule.

    Args:
        step_size_fn: A function from the update count to the step size, such as an optax
            schedule.
    """

    step_size_fn: Callable[[IntegralNumeric], ArrayLike] = field(static=True)

    def __call__(self, count: IntegralNumeric) -> ArrayLike:
        return self.step_size_fn(count)


ScalarOrSchedule = float | JaxArray | Schedule
