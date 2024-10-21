from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING

from dateutil.relativedelta import relativedelta

from ..utils import isclose_abs
from .types import DateFrequency

if TYPE_CHECKING:
    from mizani.typing import (
        DatetimeBreaksUnits,
    )


SECONDS_PER_DAY = 24 * 60 * 60
SECONDS_PER_HOUR = 60 * 60
ONE_DAY = timedelta(days=1)
ONE_WEEK = timedelta(days=7)
ONE_MONTH = relativedelta(months=1)
ONE_YEAR = relativedelta(years=1)
ONE_HOUR = timedelta(hours=1)
ONE_MINUTE = timedelta(minutes=1)
ONE_SECOND = timedelta(seconds=1)
DF = DateFrequency


@dataclass
class Interval:
    start: datetime
    end: datetime

    def __post_init__(self):
        if isinstance(self.start, date):
            self.start = datetime.fromisoformat(self.start.isoformat())

        if isinstance(self.end, date):
            self.end = datetime.fromisoformat(self.end.isoformat())

        self._delta = relativedelta(self.end, self.start)
        self._tdelta = self.end - self.start

    @property
    def y(self) -> int:
        """
        Years
        """
        return self._delta.years

    @property
    def y_wide(self) -> int:
        """
        Years (enclosing the original)
        """
        return Interval(*self.limits_year()).y

    @property
    def M(self) -> int:
        """
        Months
        """
        return self.y * 12 + self._delta.months

    @property
    def M_wide(self) -> int:
        """
        Months (enclosing the original)
        """
        return Interval(*self.limits_month()).M

    @property
    def w(self) -> int:
        """
        Weeks
        """
        return self._tdelta.days // 7

    @property
    def w_wide(self) -> int:
        """
        Weeks (enclosing the original)
        """
        return Interval(*self.limits_week()).w

    @property
    def d(self) -> int:
        """
        Days
        """
        return self._tdelta.days

    @property
    def d_wide(self) -> int:
        """
        Days (enclosing the original)
        """
        return Interval(*self.limits_day()).d

    @property
    def h(self) -> int:
        """
        Hours
        """
        return self.d * 24 + self._delta.hours

    @property
    def h_wide(self) -> int:
        """
        Hours (enclosing the original)
        """
        return Interval(*self.limits_hour()).h

    @property
    def m(self) -> int:
        """
        Minutes
        """
        return self.h * 60 + self._delta.minutes

    @property
    def m_wide(self) -> int:
        """
        Minutes (enclosing the original)
        """
        return Interval(*self.limits_minute()).m

    @property
    def s(self) -> int:
        """
        Seconds
        """
        return math.floor(self._tdelta.total_seconds())

    @property
    def u(self) -> int:
        """
        Microseconds
        """
        return math.floor(self._tdelta.total_seconds() * 1e6)

    @property
    def limits(self) -> tuple[datetime, datetime]:
        return self.start, self.end

    def limits_year(self) -> tuple[datetime, datetime]:
        return floor_year(self.start), ceil_year(self.end)

    def limits_month(self) -> tuple[datetime, datetime]:
        return round_month(self.start), round_month(self.end)

    def limits_week(self) -> tuple[datetime, datetime]:
        return floor_week(self.start), ceil_week(self.end)

    def limits_day(self) -> tuple[datetime, datetime]:
        return floor_day(self.start), ceil_day(self.end)

    def limits_hour(self) -> tuple[datetime, datetime]:
        return floor_hour(self.start), ceil_hour(self.end)

    def limits_minute(self) -> tuple[datetime, datetime]:
        return floor_minute(self.start), ceil_minute(self.end)

    def limits_second(self) -> tuple[datetime, datetime]:
        return floor_second(self.start), ceil_second(self.end)

    def limits_for_frequency(
        self, freq: DateFrequency
    ) -> tuple[datetime, datetime]:
        lookup = {
            DF.YEARLY: self.limits_year,
            DF.MONTHLY: self.limits_month,
            DF.DAILY: self.limits_day,
            DF.HOURLY: self.limits_hour,
            DF.MINUTELY: self.limits_minute,
            DF.SECONDLY: self.limits_second,
        }
        try:
            return lookup[freq]()
        except KeyError:
            return self.limits


def floor_year(d: datetime) -> datetime:
    """
    Round down to the start of the year
    """
    return d.min.replace(d.year)


def ceil_year(d: datetime) -> datetime:
    """
    Round up to start of the year
    """
    _d_floor = floor_year(d)
    if d == _d_floor:
        # Already at the start of the year
        return d
    else:
        return _d_floor + ONE_YEAR


def floor_mid_year(d: datetime) -> datetime:
    """
    Round down half a year
    """
    _d_floor = floor_year(d)
    if d.month < 7:
        return _d_floor.replace(month=1)
    else:
        return _d_floor.replace(month=7)


def ceil_mid_year(d: datetime) -> datetime:
    """
    Round up half a year
    """
    _d_floor = floor_year(d)
    if d == _d_floor:
        return d
    elif d.month < 7:
        return _d_floor.replace(month=7)
    else:
        return _d_floor + ONE_YEAR


def floor_month(d: datetime) -> datetime:
    """
    Round down to the start of the month
    """
    return d.min.replace(d.year, d.month, 1)


def ceil_month(d: datetime) -> datetime:
    """
    Round up to the start of the next month
    """
    _d_floor = floor_month(d)
    if d == _d_floor:
        return d

    return _d_floor + ONE_MONTH


def round_month(d: datetime) -> datetime:
    """
    Round date to the "nearest" start of the month
    """
    return floor_month(d) if d.day <= 15 else ceil_month(d)


def floor_week(d: datetime) -> datetime:
    """
    Round down to the start of the week

    Week start on are on 1st, 8th, 15th and 22nd
    day of the month
    """
    if d.day < 8:
        day = 1
    elif d.day < 15:
        day = 8
    elif d.day < 22:
        day = 15
    else:
        day = 22
    return d.min.replace(d.year, d.month, day)


def ceil_week(d: datetime) -> datetime:
    """
    Round up to the start of the next month

    Week start on are on 1st, 8th, 15th and 22nd
    day of the month
    """
    _d_floor = floor_week(d)
    if d == _d_floor:
        return d

    if d.day >= 22:
        return d.min.replace(d.year, d.month, d.day) + ONE_WEEK

    return _d_floor + ONE_WEEK


def floor_day(d: datetime) -> datetime:
    """
    Round down to the start of the day
    """
    return (
        d.min.replace(d.year, d.month, d.day, tzinfo=d.tzinfo)
        if has_time(d)
        else d
    )


def ceil_day(d: datetime) -> datetime:
    """
    Round up to the start of the next day
    """
    return floor_day(d) + ONE_DAY if has_time(d) else d


def floor_hour(d: datetime) -> datetime:
    """
    Round down to the start of the hour
    """
    if at_the_hour(d):
        return d
    return floor_day(d).replace(hour=d.hour)


def ceil_hour(d: datetime) -> datetime:
    """
    Round up to the start of the next hour
    """
    if at_the_hour(d):
        return d
    return floor_hour(d) + ONE_HOUR


def floor_minute(d: datetime) -> datetime:
    """
    Round down to the start of the minute
    """
    if at_the_minute(d):
        return d
    return floor_hour(d).replace(minute=d.minute)


def ceil_minute(d: datetime) -> datetime:
    """
    Round up to the start of the next minute
    """
    if at_the_minute(d):
        return d
    return floor_minute(d) + ONE_MINUTE


def floor_second(d: datetime) -> datetime:
    """
    Round down to the start of the second
    """
    if at_the_second(d):
        return d
    return floor_minute(d).replace(second=d.second)


def ceil_second(d: datetime) -> datetime:
    """
    Round up to the start of the next minute
    """
    if at_the_second(d):
        return d
    return floor_second(d) + ONE_SECOND


def has_time(d: datetime) -> bool:
    """
    Return True if the time of datetime is not 00:00:00 (midnight)
    """
    return d.time() != time.min


def at_the_hour(d: datetime) -> bool:
    """
    Return True if the time of datetime is at the hour mark
    """
    t = d.time()
    return t.minute == 0 and t.second == 0 and t.microsecond == 0


def at_the_minute(d: datetime) -> bool:
    """
    Return True if the time of datetime is at the minute mark
    """
    t = d.time()
    return t.second == 0 and t.microsecond == 0


def at_the_second(d: datetime) -> bool:
    """
    Return True if the time of datetime is at the second mark
    """
    return d.time().microsecond == 0


def align_limits(limits: tuple[int, int], width: float) -> tuple[float, float]:
    """
    Return limits so that breaks should be multiples of the width

    The new limits are equal or contain the original limits
    """
    low, high = limits

    l, m = divmod(low, width)
    if isclose_abs(m / width, 1):
        l += 1

    h, m = divmod(high, width)
    if not isclose_abs(m / width, 0):
        h += 1

    return l * width, h * width


def shift_limits_down(
    candidate_limits: tuple[int, int],
    original_limits: tuple[int, int],
    width: int,
) -> tuple[int, int]:
    """
    Shift candidate limits down so that they can be a multiple of width

    If the shift would exclude any of the original_limits (high),
    candidate limits are returned.

    The goal of this function is to convert abitrary limits into "nicer"
    ones.
    """
    low, high = candidate_limits
    low_orig, high_orig = original_limits

    l, m = divmod(low, width)
    if isclose_abs(m / width, 1):  # pragma: no cover
        l += 1

    low_new = l * width
    diff = low - low_new
    high_new = high - diff

    if high_orig <= high_new:
        return low_new, high_new

    return candidate_limits


def expand_datetime_limits(
    limits: tuple[datetime, datetime],
    width: int,
    units: DatetimeBreaksUnits,
) -> tuple[datetime, datetime]:
    ival = Interval(*limits)
    if units == "year":
        start, end = ival.limits_year()
        span = ival.y_wide
    elif units == "month":
        start, end = ival.limits_month()
        span = ival.M_wide
    elif units == "week":
        start, end = ival.limits_week()
        span = ival.w_wide
    elif units == "day":
        start, end = ival.limits_day()
        span = ival.d_wide
    elif units == "hour":
        start, end = ival.limits_hour()
        span = ival.h_wide
    elif units == "minute":
        start, end = ival.limits_minute()
        span = ival.m_wide
    elif units == "second":
        start, end = ival.limits_second()
        span = ival.s
    else:
        return limits

    new_span = math.ceil(span / width) * width
    if units == "week":
        units, new_span = "day", new_span * 7

    end = start + relativedelta(None, None, **{f"{units}s": new_span})

    if units == "year":
        limits_orig = limits[0].year, limits[1].year
        y1, y2 = shift_limits_down((start.year, end.year), limits_orig, width)
        start = start.replace(y1)
        end = end.replace(y2)

    return start, end
