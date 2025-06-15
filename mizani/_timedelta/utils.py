from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Sized, cast, overload

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

if TYPE_CHECKING:
    from typing import Sequence

    from mizani.typing import (
        FloatArrayLike,
        NDArrayFloat,
        Timedelta,
        TimedeltaArrayLike,
        TimedeltaOffset,
        TimeIntervalSIUnits,
        TimeIntervalUnits,
    )


__all__ = (
    "as_timedelta",
    "parse_timedelta_width",
)

SECONDS_PER_DAY = 24 * 60 * 60
MICROSECONDS_PER_DAY = SECONDS_PER_DAY * (10**6)
SI_LOOKUP: dict[str, TimeIntervalSIUnits] = {
    # plural
    "nanoseconds": "ns",
    "microseconds": "us",
    "milliseconds": "ms",
    "seconds": "s",
    "minutes": "min",
    "hours": "h",
    "days": "d",
    "weeks": "weeks",
    "months": "mon",
    "years": "Y",
    # singular
    "nanosecond": "ns",
    "microsecond": "us",
    "millisecond": "ms",
    "second": "s",
    "minute": "min",
    "hour": "h",
    "day": "d",
    "week": "weeks",
    "year": "Y",
    # identity
    "ns": "ns",
    "us": "us",
    "ms": "ms",
    "s": "s",
    "min": "min",
    "h": "h",
    "d": "d",
    "Y": "Y",
}

SI_LOOKUP_INV: dict[str, TimeIntervalUnits] = {
    # si
    "ns": "nanoseconds",
    "us": "microseconds",
    "ms": "milliseconds",
    "s": "seconds",
    "min": "minutes",
    "h": "hours",
    "d": "days",
    "weeks": "weeks",
    "mon": "months",
    "Y": "years",
    # singular
    "nanosecond": "nanoseconds",
    "microsecond": "microseconds",
    "millisecond": "milliseconds",
    "second": "seconds",
    "minute": "minutes",
    "hour": "hours",
    "day": "days",
    "week": "weeks",
    "month": "months",
    "year": "years",
    # identity
    "nanoseconds": "nanoseconds",
    "microseconds": "microseconds",
    "milliseconds": "milliseconds",
    "seconds": "seconds",
    "minutes": "minutes",
    "hours": "hours",
    "days": "days",
    # "weeks": "weeks",
    "months": "months",
    "years": "years",
}

# NOTE: We only deal with timedelta and pd.Timedelta


@overload
def timedelta_to_num(x: TimedeltaArrayLike) -> NDArrayFloat: ...


@overload
def timedelta_to_num(x: Timedelta) -> float: ...


def timedelta_to_num(
    x: TimedeltaArrayLike | Timedelta,
) -> NDArrayFloat | float:
    """
    Convert any timedelta to days

    This function gives us a numeric representation a timedelta that
    we can add/subtract from the numeric representation of datetimes.
    """
    _x = x if (sized := isinstance(x, Sized)) else pd.Series([x])

    if not len(_x):
        return np.array([], dtype=float)

    res: NDArrayFloat = np.array(
        [td.total_seconds() / SECONDS_PER_DAY for td in _x]
    )
    return res if sized else res[0]


def num_to_timedelta(x: FloatArrayLike) -> Sequence[pd.Timedelta]:
    """
    Convert any float array to numpy datetime64 array

    Returns pd.Timedelta because they have a larger range than
    datetime.timedelta.
    """
    return tuple(pd.Timedelta(days=val) for val in x)


def timedelta_to_microseconds(x) -> int:
    """
    Convert timedelta to microseconds
    """
    return int(x.total_seconds() * 1_000_000)


def parse_timedelta_width(width: str) -> tuple[TimeIntervalUnits, int]:
    """
    Split a width spec into the interval and the units

    Parameters
    ----------
    width :
        String to parse
    """
    interval, units = width.strip().lower().split()
    if units in ("sec", "secs"):
        units = "seconds"
    elif units in ("min", "mins"):
        units = "minutes"
    units = cast("TimeIntervalUnits", f"{units.rstrip('s')}s")
    return units, int(interval)


def as_timedelta(obj: TimedeltaOffset) -> timedelta:
    """
    Convert time difference specification to a timedelta

    Parameters
    ----------
    obj :
        Specification that can be converted to a relativedelta.

        - If a `Sequence`, it is of the form
          `("[+-]<number> <units>", "[+-]<number> <units>", ...)`
          e.g. `("1 week", "2 days", ...)`.
        - If a `str`, it is of the form `"[+-]<number> <units>"`
          e.g.`"3 hours"`.
        - If `None`, return 0 relativedelta.

        So `"3 weeks"` is equivalent to `("3 weeks",)`
    """
    if obj is None:
        return timedelta()
    elif isinstance(obj, timedelta):
        return obj
    elif isinstance(obj, relativedelta):
        if obj.months or obj.years:
            raise ValueError(
                "A relativedelta with years and months cannot "
                "be converted to a timedelta."
            )
        return timedelta(
            microseconds=obj.microseconds,
            seconds=obj.seconds,
            minutes=obj.minutes,
            hours=obj.hours,
            days=obj.days,
            weeks=obj.weeks,
        )
    elif isinstance(obj, str):
        obj = (obj,)

    kwargs = dict(parse_timedelta_width(width) for width in obj)
    return timedelta(**kwargs)
