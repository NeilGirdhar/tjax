"""Filters that summarize the history of a signal in a fixed-size state."""

from ._src.filtering.averages import LeakyAverage, LeakyAverageState, LeakyMoments
from ._src.filtering.delay_line import DelayLine
from ._src.filtering.erlang import ErlangChain
from ._src.filtering.exponential_bank import ExponentialBank
from ._src.filtering.filter import Filter, StagedFilter
from ._src.filtering.leaky_integral import LeakyIntegral
from ._src.filtering.linear import LinearFilter

__all__ = [
    "DelayLine",
    "ErlangChain",
    "ExponentialBank",
    "Filter",
    "LeakyAverage",
    "LeakyAverageState",
    "LeakyIntegral",
    "LeakyMoments",
    "LinearFilter",
    "StagedFilter",
]
