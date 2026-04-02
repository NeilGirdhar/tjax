from __future__ import annotations

import logging
from contextlib import AbstractContextManager
from time import perf_counter_ns
from types import TracebackType
from typing import Self, override

_metric_factor = 1000
log = logging.getLogger(__name__)


class Timer(AbstractContextManager["Timer"]):
    """A context manager that measures wall-clock time and logs the result.

    On exit the elapsed time is logged at INFO level using the message
    ``"{final_string} took {elapsed}"``.

    Example::

        with Timer("model forward pass"):
            y = model(x)
    """

    @override
    def __init__(self, final_string: str, precision: int = 3) -> None:
        super().__init__()
        self.start = 0
        self.end = 0
        self.final_string = final_string
        self.precision = precision

    @override
    def __enter__(self) -> Self:
        _ = super().__enter__()
        self.start = perf_counter_ns()
        return self

    @override
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
        /,
    ) -> None:
        super().__exit__(exc_type, exc_val, exc_tb)  # pyright: ignore
        self.end = perf_counter_ns()
        log.info(f"{self.final_string} took {self.elapsed_str()}")

    def elapsed_ns(self) -> int:
        """Return the elapsed time in nanoseconds."""
        return self.end - self.start

    def elapsed_str(self) -> str:
        """Return the elapsed time as a human-readable string."""
        return display_time(self.elapsed_ns(), precision=self.precision)


def display_time(elapsed_ns: float, *, precision: int = 3) -> str:
    """Format a duration given in nanoseconds as a human-readable string.

    The value is automatically scaled to the largest unit (ns, μs, ms, or s)
    that keeps it >= 1, and formatted with ``precision`` significant digits.

    Args:
        elapsed_ns: Duration in nanoseconds.
        precision: Number of significant digits in the formatted output.
    """
    elapsed = elapsed_ns
    for unit in ["n", "μ", "m"]:  # noqa: B007
        if elapsed < _metric_factor:
            break
        elapsed /= _metric_factor
    else:
        unit = ""
    if elapsed >= 1:
        precision = max(0, precision - len(str(int(elapsed))))
    return f"{elapsed:.{precision}f} {unit}s"
