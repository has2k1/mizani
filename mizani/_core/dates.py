from __future__ import annotations

import math
import sys
import typing
from dataclasses import dataclass
from datetime import datetime, timedelta, tzinfo
from enum import IntEnum
from typing import overload
from zoneinfo import ZoneInfo

import dateutil.rrule as rr
import numpy as np
from dateutil.relativedelta import relativedelta
from dateutil.rrule import rrule

from ..utils import get_timezone

if typing.TYPE_CHECKING:
    from typing import Generator, Optional, Sequence

    from mizani.typing import (
        Datetime,
        DatetimeBreaksUnits,
        FloatVector,
        NDArrayDatetime,
        NDArrayFloat,
        SeqDatetime,
        TupleFloat2,
        TupleInt2,
        TzInfo,
    )

ABS_TOL = 1e-10  # Absolute Tolerance
EPSILON = sys.float_info.epsilon

EPOCH = datetime(1970, 1, 1, tzinfo=None)
EPOCH64 = np.datetime64("1970", "Y")
SECONDS_PER_DAY = 24 * 60 * 60
MICROSECONDS_PER_DAY = SECONDS_PER_DAY * (10**6)
NaT_int = np.datetime64("NaT").astype(np.int64)
MIN_DATETIME64 = np.datetime64("0001-01-01")
MAX_DATETIME64 = np.datetime64("10000-01-01")
UTC = ZoneInfo("UTC")


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


def get_tzinfo(tz: Optional[str | TzInfo] = None) -> TzInfo | None:
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
def datetime_to_num(x: SeqDatetime) -> NDArrayFloat:
    ...


@overload
def datetime_to_num(x: Datetime) -> float:
    ...


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
    x: FloatVector, tz: Optional[str | TzInfo] = None
) -> NDArrayDatetime:
    """
    Convert any float array to numpy datetime64 array
    """
    tz = get_tzinfo(tz) or UTC
    return _from_ordinalf_np_vectorized(x, tz)


class DateFrequency(IntEnum):
    """
    Date Frequency

    Matching the dateutils constants
    """

    __order__ = "YEARLY MONTHLY DAILY HOURLY MINUTELY SECONDLY MICROSECONDLY"
    YEARLY = rr.YEARLY
    MONTHLY = rr.MONTHLY
    DAILY = rr.DAILY
    HOURLY = rr.HOURLY
    MINUTELY = rr.MINUTELY
    SECONDLY = rr.SECONDLY
    MICROSECONDLY = SECONDLY + 1


DF = DateFrequency


@dataclass
class date_breaks_info:
    """
    Information required to generate sequence of date breaks
    """

    frequency: DateFrequency
    n: int
    width: int
    start: datetime
    until: datetime
    tz: TzInfo | None


WIDTHS: dict[DateFrequency, Sequence[int]] = {
    DF.YEARLY: (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000),
    DF.MONTHLY: (1, 2, 3, 4, 6),
    DF.DAILY: (1, 2, 4, 7, 14, 21),
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


def _isclose_abs(a: float, b: float, tol: float = ABS_TOL) -> bool:
    """
    Return True if a and b are close given the absolute tolerance
    """
    return math.isclose(a, b, rel_tol=0, abs_tol=ABS_TOL)


def _align_limits(limits: TupleInt2, width: float) -> TupleFloat2:
    """
    Return limits so that breaks should be multiples of the width

    The new limits are equal or contain the original limits
    """
    low, high = limits

    l, m = divmod(low, width)
    if _isclose_abs(m / width, 1):
        l += 1

    h, m = divmod(high, width)
    if not _isclose_abs(m / width, 0):
        h += 1

    return l * width, h * width


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

    low, high = limits
    delta = relativedelta(high, low)
    tdelta = high - low

    # Calculate durations in all possible units
    # The durations round downwards
    y = delta.years
    M = y * 12 + delta.months
    d = tdelta.days
    h = d * 24 + delta.hours
    m = h * 60 + delta.minutes
    s = math.floor(tdelta.total_seconds())
    u = math.floor(tdelta.total_seconds() * 1e6)

    # Widen the duration for the year to include
    limits_year = _floor_year(low), _ceil_year(high)
    y_wide = limits_year[1].year - limits_year[0].year
    unit_durations = (y_wide, M, d, h, m, s, u)

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
            else:
                continue
            break
        else:
            continue
        break

    num_breaks = duration // break_width

    if freq == DateFrequency.YEARLY:
        limits = limits_year

    res = date_breaks_info(
        freq,
        num_breaks,
        break_width,
        start=limits[0].replace(tzinfo=tz),
        until=limits[1].replace(tzinfo=tz),
        tz=tz,
    )
    return res


def calculate_date_breaks_auto(limits, n: int = 5) -> Sequence[datetime]:
    """
    Calcuate date breaks using appropriate units
    """
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
    limits,
    units: DatetimeBreaksUnits,
    width: int,
    max_breaks: Optional[int] = None,
    tz: Optional[TzInfo] = None,
) -> Sequence[datetime]:
    """
    Calcuate date breaks using appropriate units
    """
    timely_name = f"{units.upper()}LY"
    if timely_name in ("DAYLY", "WEEKLY"):
        timely_name = "DAILY"

    freq = getattr(DF, timely_name)

    # Appropriate start and end dates
    low, high = limits
    delta = relativedelta(high, high)
    start = low - delta
    until = high + delta

    info = date_breaks_info(
        freq,
        n=-1,
        width=width,
        start=start,
        until=until,
        tz=tz,
    )

    lookup = {
        "year": yearly_breaks,
        "month": monthly_breaks,
        "week": weekly_breaks,
        "day": daily_breaks,
        "hour": hourly_breaks,
        "minute": minutely_breaks,
        "second": secondly_breaks,
        "microsecond": microsecondly_breaks,
    }
    return lookup[units](info)


def yearly_breaks(info: date_breaks_info) -> Sequence[datetime]:
    """
    Calculate yearly breaks
    """
    # New limits so that breaks fall on multiples of
    # the width
    limits = info.start.year, info.until.year
    l, h = _align_limits(limits, info.width)
    l, h = math.floor(l), math.ceil(h)

    if isinstance(info.start, datetime):
        _replace_d = {
            "month": 1,
            "day": 1,
            "hour": 0,
            "minute": 0,
            "second": 0,
            "tzinfo": info.tz,
        }
    else:  # date object
        _replace_d = {"month": 1, "day": 1}

    start = info.start.replace(year=l, **_replace_d)  # type: ignore
    until = info.until.replace(year=h, **_replace_d)  # type: ignore
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
        bymonth=range(1, 13, info.width),
    )
    breaks = r.between(info.start, info.until, True)
    if not len(breaks):
        start = _floor_mid_year(info.start)
        until = _ceil_mid_year(info.until)
        r = rrule(
            info.frequency,
            interval=info.width,
            dtstart=start,
            until=until,
            bymonth=range(1, 13, info.width),
        )
        breaks = list(r)
    return breaks


def weekly_breaks(info: date_breaks_info) -> Sequence[datetime]:
    """
    Calculate weekly breaks
    """
    info.width = info.width * 7
    return daily_breaks(info)


def daily_breaks(info: date_breaks_info) -> Sequence[datetime]:
    """
    Calculate daily breaks
    """
    if info.width == 7:
        bymonthday = (1, 8, 15, 22)
    elif info.width == 14:
        bymonthday = (1, 15)
    else:
        bymonthday = range(1, 31, info.width)

    r = rrule(
        info.frequency,
        interval=1,
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
    nmin: float
    nmax: float
    width = info.width

    nmin, nmax = datetime_to_num((info.start, info.until))
    day0: float = np.floor(nmin)
    umax = (nmax - day0) * MICROSECONDS_PER_DAY
    umin = (nmin - day0) * MICROSECONDS_PER_DAY

    # Ensure max is a multiple of the width
    width = info.width
    h, m = divmod(umax, width)
    if not _isclose_abs(m / width, 0):
        h += 1
    umax = h * width

    # Generate breaks at the multiples of the width
    n = (umax - umin + 0.001 * width) // width
    ubreaks = umin - width + np.arange(n + 3) * width
    breaks = day0 + ubreaks / MICROSECONDS_PER_DAY
    return num_to_datetime(breaks, info.tz)


def _floor_year(d: datetime) -> datetime:
    """
    Round down to the start of the year
    """
    return d.min.replace(d.year)


def _ceil_year(d: datetime) -> datetime:
    """
    Round up to start of the year
    """
    _d = _floor_year(d)
    if _d == d:
        # Already at the start of the year
        return _d
    else:
        return _d.replace(d.year + 1)


def _floor_mid_year(d: datetime) -> datetime:
    """
    Round down half a year
    """
    _d = _floor_year(d)
    if d.month < 7:
        return _d.replace(month=1)
    else:
        return _d.replace(month=7)


def _ceil_mid_year(d: datetime) -> datetime:
    """
    Round up half a year
    """
    _d = _floor_year(d)
    if _d == d:
        return _d
    elif d.month < 7:
        return _d.replace(month=7)
    else:
        return _d.replace(year=_d.year + 1, month=1)
