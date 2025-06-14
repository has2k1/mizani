from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import TYPE_CHECKING

from dateutil.relativedelta import relativedelta

from .types import DateTimeRounder

if TYPE_CHECKING:
    pass


ONE_DAY = timedelta(days=1)
ONE_WEEK = timedelta(days=7)
ONE_MONTH = relativedelta(months=1)
ONE_YEAR = relativedelta(years=1)
ONE_HOUR = timedelta(hours=1)
ONE_MINUTE = timedelta(minutes=1)
ONE_SECOND = timedelta(seconds=1)


class microseconds(DateTimeRounder):
    """
    Round by microsecond
    """

    @classmethod
    def floor(cls, d: datetime) -> datetime:
        """
        Round down to the start of the second
        """
        return d

    @classmethod
    def ceil(cls, d: datetime) -> datetime:
        """
        Round up to the next second
        """
        return d


class seconds(DateTimeRounder):
    """
    Round by second
    """

    @classmethod
    def floor(cls, d: datetime) -> datetime:
        """
        Round down to the start of the second
        """
        if at_the_second(d):
            return d
        return minutes.floor(d).replace(second=d.second)

    @classmethod
    def ceil(cls, d: datetime) -> datetime:
        """
        Round up to the next second
        """
        if at_the_second(d):
            return d
        return cls.floor(d) + ONE_SECOND


class minutes(DateTimeRounder):
    """
    Round by minute
    """

    @classmethod
    def floor(cls, d: datetime) -> datetime:
        """
        Round down to the start of the minute
        """
        if at_the_minute(d):
            return d
        return hours.floor(d).replace(minute=d.minute)

    @classmethod
    def ceil(cls, d: datetime) -> datetime:
        """
        Round up to the start of the next minute
        """
        if at_the_minute(d):
            return d
        return cls.floor(d) + ONE_MINUTE


class hours(DateTimeRounder):
    """
    Round by hour
    """

    @classmethod
    def floor(cls, d: datetime) -> datetime:
        """
        Round down to the start of the hour
        """
        if at_the_hour(d):
            return d
        return days.floor(d).replace(hour=d.hour)

    @classmethod
    def ceil(cls, d: datetime) -> datetime:
        """
        Round up to the start of the next hour
        """
        if at_the_hour(d):
            return d
        return cls.floor(d) + ONE_HOUR


class days(DateTimeRounder):
    """
    Round by day
    """

    @classmethod
    def floor(cls, d: datetime) -> datetime:
        """
        Round down to the start of the day
        """
        return datetime(d.year, d.month, d.day, tzinfo=d.tzinfo)

    @classmethod
    def ceil(cls, d: datetime) -> datetime:
        """
        Round up to the start of the next day
        """
        return cls.floor(d) + ONE_DAY if has_time(d) else d


class weeks(DateTimeRounder):
    """
    Round by week
    """

    @classmethod
    def floor(cls, d: datetime) -> datetime:
        """
        Round down to the start (Monday) of the week
        """
        return days.floor(d) - timedelta(days=d.weekday())

    @classmethod
    def ceil(cls, d: datetime) -> datetime:
        """
        Round up to the start of the next week
        """
        d = days.ceil(d)
        return d + timedelta(days=(7 - d.weekday()) % 7)


class months(DateTimeRounder):
    """
    Round by month
    """

    @classmethod
    def floor(cls, d: datetime) -> datetime:
        """
        Round down to the start of the month
        """
        return datetime(d.year, d.month, 1, tzinfo=d.tzinfo)

    @classmethod
    def ceil(cls, d: datetime) -> datetime:
        """
        Round up to the start of the next month
        """
        floor = cls.floor(d)
        return d if d == floor else floor + ONE_MONTH


class years(DateTimeRounder):
    """
    Round by year
    """

    @classmethod
    def floor(cls, d: datetime) -> datetime:
        """
        Round down to the start of the year
        """
        return datetime(d.year, 1, 1, tzinfo=d.tzinfo)

    @classmethod
    def ceil(cls, d: datetime) -> datetime:
        """
        Round up to start of the year
        """
        floor = cls.floor(d)
        return d if d == floor else floor + ONE_YEAR


class decades(DateTimeRounder):
    """
    Round by decade
    """

    @classmethod
    def floor(cls, d: datetime) -> datetime:
        """
        Round down to the start of the decade
        """
        return datetime(d.year - d.year % 10, 1, 1, tzinfo=d.tzinfo)

    @classmethod
    def ceil(cls, d: datetime) -> datetime:
        """
        Round up to the start of the next decade
        """
        floor = cls.floor(d)
        return d if d == floor else floor.replace(floor.year + 10)


class centurys(DateTimeRounder):
    """
    Round by century
    """

    @classmethod
    def floor(cls, d: datetime) -> datetime:
        """
        Round down to the start of the century
        """
        return datetime(d.year - d.year % 100, 1, 1, tzinfo=d.tzinfo)

    @classmethod
    def ceil(cls, d: datetime) -> datetime:
        """
        Round up to the start of the next decade
        """
        floor = cls.floor(d)
        return d if d == floor else floor.replace(floor.year + 100)


def has_time(d: datetime) -> bool:
    """
    Return True if the time of datetime is not 00:00:00 (midnight)
    """
    return d.time() != time.min


def at_the_second(d: datetime) -> bool:
    """
    Return True if the time of datetime is at the second mark
    """
    return d.time().microsecond == 0


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
