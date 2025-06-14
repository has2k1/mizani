from __future__ import annotations

from datetime import datetime, timedelta
from functools import cached_property
from typing import TYPE_CHECKING, cast

import dateutil.rrule as rr
import numpy as np
from dateutil.rrule import rrule

from mizani._datetime.utils import (
    PerSecond,
    as_relativedelta,
    parse_datetime_width,
)

from ..utils import forward_fill, round_any, trim_breaks

if TYPE_CHECKING:
    from typing import Sequence

    from mizani.typing import (
        DatetimeOffset,
        DatetimeWidthUnits,
    )

    from .types import DateTimeRounder

__all__ = (
    "by_n",
    "by_width",
)

MICROSECONDLY = 7
per_sec = PerSecond()

FREQ_LOOKUP: dict[DatetimeWidthUnits, int] = {
    "microseconds": MICROSECONDLY,
    "seconds": rr.SECONDLY,
    "minutes": rr.MINUTELY,
    "hours": rr.HOURLY,
    "days": rr.DAILY,
    "weeks": rr.WEEKLY,
    "months": rr.MONTHLY,
    "years": rr.YEARLY,
}

FREQ_LOOKUP_INV: dict[int, DatetimeWidthUnits] = {
    code: unit for unit, code in FREQ_LOOKUP.items()
}


def _to_seconds(
    steps: Sequence[int | float],
    seconds_per_step: float,
) -> Sequence[float]:
    """
    Convert each step to seconds
    """
    return tuple(x * seconds_per_step for x in steps)


class Helper:
    """
    Helper for datetime intervals
    """

    # The preferred intervals at each of the units
    microseconds = (
        1,
        2,
        5,
        10,
        20,
        50,
        100,
        200,
        500,
        1000,
        2000,
        5000,
        10000,
        20000,
        50000,
        100000,
        200000,
        500000,
    )
    seconds = (1, 2, 5, 10, 15, 30)
    minutes = (1, 2, 5, 10, 15, 30)
    hours = (1, 2, 3, 4, 6, 12)
    days = (1, 2)
    weeks = (1,)
    months = (0.5, 1, 2, 3, 4, 6)
    years = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)

    @cached_property
    def intervals(self) -> Sequence[int | float]:
        """
        Intervals
        """
        return (
            *self.microseconds,
            *self.seconds,
            *self.minutes,
            *self.hours,
            *self.days,
            *self.weeks,
            *self.months,
            *self.years,
        )

    @cached_property
    def intervals_sec(self) -> Sequence[float]:
        """
        Intervals in seconds
        """
        return (
            *per_sec(self.microseconds, "us"),
            *self.seconds,
            *per_sec(self.minutes, "min"),
            *per_sec(self.hours, "h"),
            *per_sec(self.days, "d"),
            *per_sec(self.weeks, "weeks"),
            *per_sec(self.months, "mon"),
            *per_sec(self.years, "Y"),
        )

    @cached_property
    def rounders(self) -> Sequence[DateTimeRounder]:
        """
        The floor (rounding the lower limit) at each interval
        """
        from . import rounding

        lookup_unit: dict[int | float, DateTimeRounder] = {
            0: rounding.microseconds,
            1: rounding.minutes,
            per_sec(2, "min"): rounding.hours,
            per_sec(3, "h"): rounding.days,
            per_sec(1, "weeks"): rounding.weeks,
            per_sec(0.5, "mon"): rounding.months,
            per_sec(3, "mon"): rounding.years,
            per_sec(2, "Y"): rounding.decades,
            per_sec(20, "Y"): rounding.centurys,
        }
        return forward_fill([lookup_unit.get(i) for i in self.intervals_sec])

    @cached_property
    def freq(self) -> Sequence[int]:
        """
        RRule frequency at each interval
        """
        return [
            *(MICROSECONDLY,) * len(self.microseconds),
            *(rr.SECONDLY,) * len(self.seconds),
            *(rr.MINUTELY,) * len(self.minutes),
            *(rr.HOURLY,) * len(self.hours),
            *(rr.DAILY,) * len(self.days),
            *(rr.WEEKLY,) * len(self.weeks),
            *(rr.MONTHLY,) * len(self.months),
            *(rr.YEARLY,) * len(self.years),
        ]


H = Helper()


def by_n(limits: tuple[datetime, datetime], n: int = 5) -> Sequence[datetime]:
    """
    Calculate date breaks with roughly n intervals
    """
    span = (limits[1] - limits[0]).total_seconds()
    ns = span / np.array(H.intervals_sec)
    idx = cast("int", np.argmin(np.abs(ns - n)))
    freq, interval, round = H.freq[idx], H.intervals[idx], H.rounders[idx]
    units = FREQ_LOOKUP_INV[freq]

    if units == "microseconds":
        return microsecondly_breaks(limits, interval)

    bymonthday = None

    # Reinterprete any float intervals in the declarations as integers.
    # e.g. half-month intervals becomes daily interval with the 1st & 15th
    # of the month as the only permitted days.
    if isinstance(interval, float):
        if (interval, freq) == (0.5, rr.MONTHLY):
            interval = 1
            freq = rr.DAILY
            bymonthday = (1, 15)

        assert isinstance(interval, int), f"{interval=} should be an integer"

    # Add a padding the end limit so that the generated breaks include
    pad = as_relativedelta(f"{interval} {units}")
    dtstart = round.floor(limits[0])
    until = round.ceil(limits[1] + pad)
    r = rrule(freq, dtstart, interval, until=until, bymonthday=bymonthday)
    breaks = list(r)
    return trim_breaks(breaks, limits) if len(breaks) > n else breaks


def by_width(
    limits: tuple[datetime, datetime],
    width: str,
    offset: int | DatetimeOffset = None,
) -> Sequence[datetime]:
    """
    Calculate date breaks of a given width
    """
    from . import rounding

    units, interval = parse_datetime_width(width)
    if isinstance(offset, int):
        offset = f"{offset} {units}"
    offset = as_relativedelta(offset)

    if units == "microseconds":
        breaks = microsecondly_breaks(limits, interval)
    else:
        round = cast("DateTimeRounder", getattr(rounding, units))
        freq = FREQ_LOOKUP[units]
        pad = as_relativedelta(f"{interval} {units}")
        dtstart = round.floor(limits[0])
        until = round.ceil(limits[1] + pad)
        r = rrule(freq, dtstart, interval, until=until)
        breaks = trim_breaks(list(r), limits)

    return [b + offset for b in breaks]


def microsecondly_breaks(limits: tuple[datetime, datetime], interval: float):
    """
    Calculate breaks with microsecond intervals
    """
    l0, l1 = limits
    interval = cast("int", interval)

    # Round the limits to times with good microsecond values
    u0 = int(round_any(l0.microsecond, interval, np.floor))
    u1 = int(round_any(l1.microsecond, interval, np.ceil))
    start = l0.replace(microsecond=u0)
    stop = l1.replace(microsecond=u1)

    # Find the number of breaks that fits into the new span
    # and generate the sequence of breaks.
    span = (stop - start).total_seconds() * 1_000_000
    n = int(np.ceil(span / interval)) + 1
    breaks = [start + timedelta(microseconds=interval * i) for i in range(n)]
    return trim_breaks(breaks, limits)
