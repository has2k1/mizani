from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..utils import round_any, trim_breaks
from .utils import (
    SI_LOOKUP,
    SI_LOOKUP_INV,
    as_timedelta,
    parse_timedelta_width,
    timedelta_to_microseconds,
)

if TYPE_CHECKING:
    from typing import Literal, Sequence

    from mizani.typing import (
        NDArrayFloat,
        Timedelta,
        TimedeltaArrayLike,
        TimedeltaOffset,
        TimeIntervalSIUnits,
        TimeIntervalUnits,
    )

__all__ = (
    "by_n",
    "by_width",
)

SECONDS: dict[TimeIntervalSIUnits, float] = {
    "ns": 1e-9,
    "us": 1e-6,
    "ms": 1e-3,
    "s": 1,
    "min": 60,
    "h": 3600,
    "d": 24 * 3600,
    "weeks": 7 * 24 * 3600,
}

NANOSECONDS: dict[TimeIntervalSIUnits, float] = {
    "ns": 1,
    "us": 1e3,
    "ms": 1e6,
    "s": 1e9,
    "min": 60e9,
    "h": 3600e9,
    "d": 24 * 3600e9,
    "weeks": 7 * 24 * 3600e9,
}

MICROSECONDS: dict[TimeIntervalSIUnits, int] = {
    "us": 1,
    "ms": 1_000,
    "s": 1_000_000,
    "min": 1_000_000 * 60,
    "h": 1_000_000 * 60 * 60,
    "d": 1_000_000 * 60 * 60 * 24,
    "weeks": 1_000_000 * 60 * 60 * 24 * 7,
}


# This could be cleaned up, state overload?
@dataclass()
class Helper:
    """
    Helper for computing timedelta breaks
    and labels.

    How to use - breaks?

    1. Initialise with a timedelta sequence/limits.
    2. Get the scaled limits and use those to calculate
       breaks using a general purpose breaks calculating
       routine. The scaled limits are in numerical format.
    3. Convert the computed breaks from numeric into timedelta.

    See, :class:`breaks_timedelta`

    How to use - formating?

    1. Call :meth:`format_info` with the timedelta values to be
       formatted and get back a tuple of numeric values and
       the units for those values.
    2. Format the values with a general purpose formatting
       routing.

    See, :class:`~mizani.labels.label_timedelta`
    """

    x: TimedeltaArrayLike
    units: TimeIntervalUnits | TimeIntervalSIUnits | None = None

    def __post_init__(self):
        l, h = min(self.x), max(self.x)
        self.package = self.determine_package(self.x[0])
        self.limits = self.value(l), self.value(h)
        self._units: TimeIntervalSIUnits = (
            SI_LOOKUP[self.units] if self.units else self.best_units((l, h))
        )
        self.factor = self.get_scaling_factor(self._units)

    @classmethod
    def determine_package(cls, td: Timedelta) -> Literal["pandas", "cpython"]:
        if hasattr(td, "components"):
            package = "pandas"
        elif hasattr(td, "total_seconds"):
            package = "cpython"
        else:
            msg = f"{td.__class__} format not yet supported."
            raise ValueError(msg)
        return package

    @classmethod
    def format_info(
        cls,
        x: TimedeltaArrayLike,
        units: TimeIntervalUnits | TimeIntervalSIUnits | None = None,
    ) -> tuple[Sequence[int | float], TimeIntervalSIUnits]:
        ins = cls(x, units)
        return ins.timedelta_to_numeric(x), ins._units

    def best_units(self, x: TimedeltaArrayLike) -> TimeIntervalSIUnits:
        """
        Determine good units for representing a sequence of timedeltas
        """
        # Read
        #   [(0.9, 's'),
        #    (9, 'm)]
        # as, break ranges between 0.9 seconds (inclusive)
        # and 9 minutes are represented in seconds. And so on.
        ts_range = self.value(max(x)) - self.value(min(x))
        package = self.determine_package(x[0])
        if package == "pandas":
            cuts: list[tuple[float, TimeIntervalSIUnits]] = [
                (0.9, "us"),
                (0.9, "ms"),
                (0.9, "s"),
                (5, "min"),
                (6, "h"),
                (5, "d"),
                (4, "weeks"),
                (365.25, "d"),
            ]
            denomination = NANOSECONDS
            base_units = "ns"
        else:
            cuts = [
                (0.9, "s"),
                (5, "min"),
                (6, "h"),
                (5, "d"),
                (4, "weeks"),
                (365.25, "d"),
            ]
            denomination = SECONDS
            base_units = "ms"

        for size, units in reversed(cuts):
            if ts_range >= size * denomination[units]:
                return units

        return base_units

    @staticmethod
    def value(td: Timedelta) -> float:
        """
        Return the numeric value representation on a timedelta
        """
        if isinstance(td, pd.Timedelta):
            return td.value
        else:
            return td.total_seconds()

    def scaled_limits(self) -> tuple[float, float]:
        """
        Minimum and Maximum to use for computing breaks
        """
        _min = self.limits[0] / self.factor
        _max = self.limits[1] / self.factor
        return _min, _max

    def timedelta_to_numeric(
        self, timedeltas: TimedeltaArrayLike
    ) -> Sequence[int | float]:
        """
        Convert sequence of timedelta to numerics
        """
        return [self.to_numeric(td) for td in timedeltas]

    def numeric_to_timedelta(self, values: NDArrayFloat) -> TimedeltaArrayLike:
        """
        Convert sequence of numerical values to timedelta
        """
        if self.package == "pandas":
            return [
                pd.Timedelta(int(x * self.factor), unit="ns") for x in values
            ]

        else:
            units = SI_LOOKUP_INV[self._units]
            return [timedelta(**{units: x}) for x in values]

    def get_scaling_factor(self, units):
        if self.package == "pandas":
            return NANOSECONDS[units]
        else:
            return SECONDS[units]

    def to_numeric(self, td: Timedelta) -> int | float:
        """
        Convert timedelta to a number corresponding to the
        appropriate units. The appropriate units are those
        determined with the object is initialised.
        """
        if isinstance(td, pd.Timedelta):
            res = td.value / NANOSECONDS[self._units]
        else:
            res = td.total_seconds() / SECONDS[self._units]

        return int(res) if int(res) == res else float(res)


def by_n(
    limits: tuple[timedelta, timedelta], n: int = 5
) -> Sequence[timedelta]:
    """
    Calculate timedelta breaks with roughly n intervals
    """
    from mizani.breaks import breaks_extended

    if any(pd.isna(x) for x in limits):
        return []
    h = Helper(limits)
    scaled_limits = h.scaled_limits()
    scaled_breaks = breaks_extended(n, (1, 2, 5, 10))(scaled_limits)
    breaks = h.numeric_to_timedelta(scaled_breaks)
    return trim_breaks(list(breaks), limits)


def by_width(
    limits: tuple[timedelta, timedelta],
    width: str,
    offset: int | TimedeltaOffset = None,
) -> Sequence[timedelta]:
    """
    Calculate timedelta breaks of a given width
    """
    units, interval = parse_timedelta_width(width)
    interval = MICROSECONDS[SI_LOOKUP[units]] * interval

    us0 = timedelta_to_microseconds(limits[0])
    us1 = timedelta_to_microseconds(limits[1])
    us0 = int(round_any(us0, interval, np.floor))
    us1 = int(round_any(us1, interval, np.ceil))

    if isinstance(offset, int):
        offset = f"{offset} {units}"

    offset = as_timedelta(offset)
    n = int((us1 - us0) / interval) + 1
    start = timedelta(microseconds=us0)
    breaks = [start + timedelta(microseconds=interval * i) for i in range(n)]
    return [b + offset for b in trim_breaks(breaks, limits)]
