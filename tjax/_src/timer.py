from __future__ import annotations

import logging
from contextlib import AbstractContextManager
from time import perf_counter_ns
from types import TracebackType
from typing import Self, override

_metric_factor = 1000
log = logging.getLogger(__name__)


class Timer(AbstractContextManager['Timer']):
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
    def __exit__(self,
                 exc_type: type[BaseException] | None,
                 exc_val: BaseException | None,
                 exc_tb: TracebackType | None,
                 /) -> None:
        super().__exit__(exc_type, exc_val, exc_tb)  # pyright: ignore
        self.end = perf_counter_ns()
        log.info(f"{self.final_string} took {self.elapsed_str()}")

    def elapsed_ns(self) -> int:
        return self.end - self.start

    def elapsed_str(self) -> str:
        return display_time(self.elapsed_ns(), precision=self.precision)


def display_time(elapsed_ns: float, *, precision: int = 3) -> str:
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
