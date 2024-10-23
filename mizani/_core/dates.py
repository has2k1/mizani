from __future__ import annotations

import math
from collections.abc import Sized
from datetime import datetime, timedelta, tzinfo
from typing import TYPE_CHECKING, overload
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from dateutil.rrule import rrule

from ..utils import get_timezone, isclose_abs
from .date_utils import (
    Interval,
    align_limits,
    as_datetime,
    expand_datetime_limits,
)
from .types import DateFrequency, date_breaks_info

if TYPE_CHECKING:
    from typing import Generator, Sequence

    from mizani.typing import (
        Datetime,
        DatetimeBreaksUnits,
        FloatArrayLike,
        NDArrayDatetime,
        NDArrayFloat,
        SeqDatetime,
        Timedelta,
        TimedeltaArrayLike,
    )


EPOCH = datetime(1970, 1, 1, tzinfo=None)
EPOCH64 = np.datetime64("1970", "Y")
SECONDS_PER_DAY = 24 * 60 * 60
MICROSECONDS_PER_DAY = SECONDS_PER_DAY * (10**6)

NaT_int = np.datetime64("NaT").astype(np.int64)
MIN_DATETIME64 = np.datetime64("0001-01-01")
MAX_DATETIME64 = np.datetime64("10000-01-01")
UTC = ZoneInfo("UTC")
DF = DateFrequency


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


WIDTHS: dict[DateFrequency, Sequence[int]] = {
    DF.YEARLY: (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000),
    DF.MONTHLY: (1, 2, 3, 4, 6),
    DF.DAILY: (1, 2, 4, 7, 14),
    DF.HOURLY: (1, 2, 3, 4, 6, 12),
    DF.MINUTELY: (1, 5, 10, 15, 30),
    DF.SECONDLY: (1, 5, 10, 15, 30),
    DF.MICROSECONDLY: (
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
        1000000,
    ),
}

MAX_BREAKS = dict(zip(DateFrequency, (11, 12, 11, 12, 11, 11, 8)))


def _viable_freqs(
    min_breaks: int,
    unit_durations: tuple[int, int, int, int, int, int, int],
) -> Generator[tuple[DateFrequency, int], None, None]:
    """
    Find viable frequency, duration pairs

    A pair is viable if it can yeild a suitable number of breaks
    For example:
        - YEARLY frequency, 3 year unit_duration and
          8 min_breaks is not viable
        - MONTHLY frequency, 36 month unit_duration and
          8 min_breaks is viable
    """
    for freq, duration in zip(DateFrequency, unit_durations):
        max_width = WIDTHS[freq][-1]
        max_breaks = max(min_breaks, MAX_BREAKS.get(freq, 11))
        if duration <= max_width * max_breaks - 1:
            yield (freq, duration)


def calculate_date_breaks_info(
    limits: tuple[datetime, datetime], n: int = 5
) -> date_breaks_info:
    """
    Calculate information required to generate breaks

    Parameters
    ----------
    limits:
        Datetime limits for the breaks
    n:
        Desired number of breaks.
    """
    tz = get_timezone(limits)
    _max_breaks_lookup = {f: max(b, n) for f, b in MAX_BREAKS.items()}

    # Widen the duration at each granularity
    itv = Interval(*limits)
    unit_durations = (
        itv.y_wide,
        itv.M_wide,
        itv.d_wide,
        itv.h_wide,
        itv.m_wide,
        itv.s,
        itv.u,
    )
    # Search frequencies from longest (yearly) to the smallest
    # for one that would a good width between the breaks

    # Defaults
    freq = DF.YEARLY  # makes pyright happy
    break_width = 1
    duration = n

    for freq, duration in _viable_freqs(n, unit_durations):
        # Search for breaks in the range
        _max_breaks = _max_breaks_lookup[freq]
        for mb in range(n, _max_breaks + 1):
            # There are few breaks at this frequency
            # e.g. (freq=YEARLY, duration=2) but mb = 5
            if duration < mb:
                continue

            for break_width in WIDTHS[freq]:
                if duration <= break_width * mb - 1:
                    break
            else:  # pragma: no cover
                continue
            break
        else:
            continue
        break

    num_breaks = duration // break_width
    limits = itv.limits_for_frequency(freq)
    res = date_breaks_info(
        freq,
        num_breaks,
        break_width,
        start=limits[0].replace(tzinfo=tz),
        until=limits[1].replace(tzinfo=tz),
        tz=tz,
    )
    return res


def calculate_date_breaks_auto(
    limits: tuple[datetime, datetime], n: int = 5
) -> Sequence[datetime]:
    """
    Calcuate date breaks using appropriate units
    """
    limits = as_datetime(limits)
    info = calculate_date_breaks_info(limits, n=n)
    lookup = {
        DF.YEARLY: yearly_breaks,
        DF.MONTHLY: monthly_breaks,
        DF.DAILY: daily_breaks,
        DF.HOURLY: hourly_breaks,
        DF.MINUTELY: minutely_breaks,
        DF.SECONDLY: secondly_breaks,
        DF.MICROSECONDLY: microsecondly_breaks,
    }
    return lookup[info.frequency](info)


def calculate_date_breaks_byunits(
    limits: tuple[datetime, datetime],
    units: DatetimeBreaksUnits,
    width: int,
    max_breaks: int | None = None,
    tz: tzinfo | None = None,
) -> Sequence[datetime]:
    """
    Calcuate date breaks using appropriate units
    """
    timely_name = f"{units.upper()}LY"
    if timely_name in ("DAYLY", "WEEKLY"):
        timely_name = "DAILY"

    freq = getattr(DF, timely_name)

    # Appropriate start and end dates
    start, until = expand_datetime_limits(limits, width, units)

    if units == "week":
        width *= 7

    info = date_breaks_info(
        freq,
        n=-1,
        width=width,
        start=start,
        until=until,
        tz=tz,
    )

    lookup = {
        "year": rrulely_breaks,
        "month": rrulely_breaks,
        "week": rrulely_breaks,
        "day": rrulely_breaks,
        "hour": rrulely_breaks,
        "minute": rrulely_breaks,
        "second": rrulely_breaks,
        "microsecond": microsecondly_breaks,
    }
    return lookup[units](info)


def rrulely_breaks(info: date_breaks_info) -> Sequence[datetime]:
    r = rrule(
        info.frequency,
        interval=info.width,
        dtstart=info.start,
        until=info.until,
    )
    return list(r)


def yearly_breaks(info: date_breaks_info) -> Sequence[datetime]:
    """
    Calculate yearly breaks
    """
    # New limits so that breaks fall on multiples of
    # the width
    limits = info.start.year, info.until.year
    l, h = align_limits(limits, info.width)
    l, h = math.floor(l), math.ceil(h)

    _replace_d = {
        "month": 1,
        "day": 1,
        "hour": 0,
        "minute": 0,
        "second": 0,
        "tzinfo": info.tz,
    }

    start = info.start.replace(year=l, **_replace_d)
    until = info.until.replace(year=h, **_replace_d)
    r = rrule(
        info.frequency,
        interval=info.width,
        dtstart=start,
        until=until,
    )
    return list(r)


def monthly_breaks(info: date_breaks_info) -> Sequence[datetime]:
    """
    Calculate monthly breaks
    """
    r = rrule(
        info.frequency,
        interval=info.width,
        dtstart=info.start,
        until=info.until,
    )
    return list(r)


def daily_breaks(info: date_breaks_info) -> Sequence[datetime]:
    """
    Calculate daily breaks
    """
    if info.width == 7:
        interval = 1
        bymonthday = (1, 8, 15, 22)
    elif info.width == 14:
        interval = 1
        bymonthday = (1, 15)
    else:
        interval = info.width
        bymonthday = None

    r = rrule(
        info.frequency,
        interval=interval,
        dtstart=info.start,
        until=info.until,
        bymonthday=bymonthday,
    )
    return r.between(info.start, info.until, True)


def hourly_breaks(info: date_breaks_info) -> Sequence[datetime]:
    """
    Calculate hourly breaks
    """
    r = rrule(
        info.frequency,
        interval=1,
        dtstart=info.start,
        until=info.until,
        byhour=range(0, 24, info.width),
    )
    return r.between(info.start, info.until, True)


def minutely_breaks(info: date_breaks_info) -> Sequence[datetime]:
    """
    Calculate minutely breaks
    """
    r = rrule(
        info.frequency,
        interval=1,
        dtstart=info.start,
        until=info.until,
        byminute=range(0, 60, info.width),
    )
    return r.between(info.start, info.until, True)


def secondly_breaks(info: date_breaks_info) -> Sequence[datetime]:
    """
    Calculate minutely breaks
    """
    r = rrule(
        info.frequency,
        interval=1,
        dtstart=info.start,
        until=info.until,
        bysecond=range(0, 60, info.width),
    )
    return r.between(info.start, info.until, True)


def microsecondly_breaks(info: date_breaks_info) -> NDArrayDatetime:
    """
    Calculate breaks at microsecond intervals
    """
    # TODO: A little too complicated, could use some refactoring
    nmin: float
    nmax: float
    width = info.width

    nmin, nmax = datetime_to_num((info.start, info.until))
    day0: float = np.floor(nmin)

    # difference in microseconds
    umax = (nmax - day0) * MICROSECONDS_PER_DAY
    umin = (nmin - day0) * MICROSECONDS_PER_DAY

    # Ensure max is a multiple of the width
    width = info.width
    h, m = divmod(umax, width)
    if not isclose_abs(m / width, 0):
        h += 1
    umax = h * width

    # Generate breaks at the multiples of the width
    n = (umax - umin + 0.001 * width) // width
    ubreaks = umin - width + np.arange(n + 3) * width
    breaks = day0 + ubreaks / MICROSECONDS_PER_DAY
    return num_to_datetime(breaks, info.tz)
