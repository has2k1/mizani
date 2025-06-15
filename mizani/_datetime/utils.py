from __future__ import annotations

from datetime import date, datetime, timedelta, tzinfo
from typing import TYPE_CHECKING, cast, overload
from zoneinfo import ZoneInfo

import numpy as np
from dateutil.relativedelta import relativedelta

if TYPE_CHECKING:
    from typing import Sequence, TypeVar

    from mizani.typing import (
        Datetime,
        DatetimeOffset,
        DatetimeWidthUnits,
        FloatArrayLike,
        NDArrayDatetime,
        NDArrayFloat,
        SeqDatetime,
        TimeIntervalSIUnits,
    )

    TFloatSeq = TypeVar("TFloatSeq", float, Sequence[float])


EPOCH = datetime(1970, 1, 1, tzinfo=None)
EPOCH64 = np.datetime64("1970", "Y")
SECONDS_PER_DAY = 24 * 60 * 60
MICROSECONDS_PER_DAY = SECONDS_PER_DAY * (10**6)

NaT_int = np.datetime64("NaT").astype(np.int64)
MIN_DATETIME64 = np.datetime64("0001-01-01")
MAX_DATETIME64 = np.datetime64("10000-01-01")
UTC = ZoneInfo("UTC")


class PerSecond:
    """
    Convert time interval to seconds
    """

    ns = 10**-9
    us = 10**-6
    ms = 10**-3
    s = 1
    min = 60
    h = 60 * min
    d = 24 * h
    weeks = 7 * d
    Y = int(365.25 * d)
    mon = Y // 12

    def __call__(self, x: TFloatSeq, units: TimeIntervalSIUnits) -> TFloatSeq:
        """
        Convert value to from some units to seconds
        """
        s_per_unit: float = getattr(self, units)
        if isinstance(x, (int, float)):
            return x * s_per_unit
        else:
            return type(x)((value * s_per_unit for value in x))  # pyright: ignore[reportCallIssue]


def _from_ordinalf(x: float, tz: tzinfo | None) -> datetime:
    """
    Convert float array to datetime
    """
    dt64 = EPOCH64 + np.timedelta64(
        int(np.round(x * MICROSECONDS_PER_DAY)), "us"
    )
    if not (MIN_DATETIME64 < dt64 <= MAX_DATETIME64):
        raise ValueError(
            f"Date ordinal {x} converts to {dt64} (using "
            f"epoch {EPOCH}). The supported dates must be  "
            "between year 0001 and 9999."
        )

    # convert from datetime64 to datetime:
    dt: datetime = dt64.astype(object)

    # but maybe we are working in a different timezone so move.
    if tz:
        # datetime64 is always UTC:
        dt = dt.replace(tzinfo=UTC)
        dt = dt.astimezone(tz)

    # fix round off errors
    if np.abs(x) > 70 * 365:
        # if x is big, round off to nearest twenty microseconds.
        # This avoids floating point roundoff error
        ms = round(dt.microsecond / 20) * 20
        if ms == 1000000:
            dt = dt.replace(microsecond=0) + timedelta(seconds=1)
        else:
            dt = dt.replace(microsecond=ms)

    return dt


_from_ordinalf_np_vectorized = np.vectorize(_from_ordinalf, otypes="O")


def get_tzinfo(tz: str | tzinfo | None = None) -> tzinfo | None:
    """
    Generate `~datetime.tzinfo` from a string or return `~datetime.tzinfo`.

    If argument is None, return None.
    """
    if tz is None:
        return None

    if isinstance(tz, str):
        return ZoneInfo(tz)

    if isinstance(tz, tzinfo):
        return tz

    raise TypeError("tz must be string or tzinfo subclass.")


@overload
def datetime_to_num(x: SeqDatetime) -> NDArrayFloat: ...


@overload
def datetime_to_num(x: Datetime) -> float: ...


def datetime_to_num(x: SeqDatetime | Datetime) -> NDArrayFloat | float:
    """
    Convery any datetime sequence to float array
    """
    iterable = np.iterable(x)
    _x = x if iterable else [x]
    try:
        x0 = next(iter(_x))
    except StopIteration:
        return np.array([], dtype=float)

    if isinstance(x0, datetime) and x0.tzinfo:
        _x = [
            dt.astimezone(UTC).replace(tzinfo=None)  # type: ignore
            for dt in _x
        ]
    res = datetime64_to_num(np.asarray(_x, dtype="datetime64"))
    return res if iterable else res[0]


def datetime64_to_num(x: NDArrayDatetime) -> NDArrayFloat:
    """
    Convery any numpy datetime64 array to float array

    In units of days
    """
    x_secs = x.astype("datetime64[s]")
    diff_ns = (x - x_secs).astype("timedelta64[ns]")

    # In seconds + nanoseconds, in days
    res = (
        (x_secs - EPOCH64).astype(np.float64)
        + diff_ns.astype(np.float64) / 1.0e9
    ) / SECONDS_PER_DAY

    x_int = x.astype(np.int64)
    res[x_int == NaT_int] = np.nan
    return res


def num_to_datetime(
    x: FloatArrayLike, tz: str | tzinfo | None = None
) -> NDArrayDatetime:
    """
    Convert any float array to numpy datetime64 array
    """
    tz = get_tzinfo(tz) or UTC
    return _from_ordinalf_np_vectorized(x, tz)


def as_datetime(
    tup: tuple[datetime, datetime] | tuple[date, date],
) -> tuple[datetime, datetime]:
    """
    Ensure that a tuple of datetime values
    """
    l, h = tup

    if not isinstance(l, datetime):
        l = datetime.fromisoformat(l.isoformat())

    if not isinstance(h, datetime):
        h = datetime.fromisoformat(h.isoformat())

    return l, h


def as_relativedelta(obj: DatetimeOffset) -> relativedelta:
    """
    Convert time difference specification to a relativedelta

    Working with relativedelta object allows us to do date arithmentic
    involving the non-uniform intervals of months and years that cannot
    be represented with a timedelta.

    Parameters
    ----------
    obj :
        Specification that can be converted to a relativedelta.

        - If a `Sequence`, it is of the form
          `("[+-]<number> <units>", "[+-]<number> <units>", ...)`
          e.g. `("1 year", "2 months", ...)`.
        - If a `str`, it is of the form `"[+-]<number> <units>"`
          e.g.`"2 years"`.
        - If `None`, return 0 relativedelta.

        So `"3 years"` is equivalent to `("3 years",)`
    """
    if obj is None:
        return relativedelta()
    elif isinstance(obj, timedelta):
        return relativedelta() + obj
    elif isinstance(obj, relativedelta):
        return obj
    elif isinstance(obj, str):
        obj = (obj,)

    kwargs = dict(parse_datetime_width(width) for width in obj)
    return relativedelta(**kwargs)


def parse_datetime_width(width: str) -> tuple[DatetimeWidthUnits, int]:
    """
    Split a width spec into the interval and the units

    Parameters
    ----------
    width :
        String to parse
    """
    interval, units = width.strip().lower().split()
    interval = int(interval)

    if units in ("decade", "decades"):
        units, interval = "years", interval * 10
    elif units in ("century", "centurys", "centuries"):
        units, interval = "years", interval * 100
    elif units in ("sec", "secs"):
        units = "seconds"
    elif units in ("min", "mins"):
        units = "minutes"
    elif units == "milliseconds":
        units, interval = "microseconds", interval * 1000

    # Convert any singular form to plural e.g year to years
    units = units.rstrip("s")
    units = cast("DatetimeWidthUnits", f"{units}s")
    return units, interval


@overload
def dt(s: str) -> datetime: ...


@overload
def dt(s: tuple[str, str]) -> tuple[datetime, datetime]: ...


@overload
def dt(s: tuple[str, ...]) -> Sequence[datetime]: ...


def dt(s: str | tuple[str, ...]) -> datetime | Sequence[datetime]:
    """
    Create datetime from isoformat

    Creating datetime objects in isoformat is more readable and
    this function gives us a succinct way to do it.
    """
    if isinstance(s, str):
        return datetime.fromisoformat(s)
    return tuple(datetime.fromisoformat(x) for x in s)
