from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from mizani._core.date_utils import (
    align_limits,
    ceil_mid_year,
    ceil_second,
    ceil_week,
    floor_mid_year,
    floor_second,
    floor_week,
)
from mizani._core.dates import (
    datetime_to_num,
    get_tzinfo,
    num_to_datetime,
    timedelta_to_num,
)


def test_tzinfo():
    tz = ZoneInfo("Africa/Kampala")
    assert get_tzinfo("Africa/Kampala") == tz
    assert get_tzinfo(tz) is tz
    with pytest.raises(TypeError):
        assert get_tzinfo(10)  # type: ignore


def test_floor_mid_year():
    d1 = datetime(2022, 3, 1)
    d2 = datetime(2022, 11, 9)
    assert floor_mid_year(d1) == datetime(2022, 1, 1)
    assert floor_mid_year(d2) == datetime(2022, 7, 1)


def test_ceil_mid_year():
    d1 = datetime(2022, 1, 1)
    d2 = datetime(2022, 1, 2)
    d3 = datetime(2022, 8, 2)
    assert ceil_mid_year(d1) == datetime(2022, 1, 1)
    assert ceil_mid_year(d2) == datetime(2022, 7, 1)
    assert ceil_mid_year(d3) == datetime(2023, 1, 1)


def test_floor_week():
    d1 = datetime(2000, 1, 11)
    d2 = datetime(2000, 8, 21)
    assert floor_week(d1) == datetime(2000, 1, 8)
    assert floor_week(d2) == datetime(2000, 8, 15)


def test_ceil_week():
    d1 = datetime(2000, 1, 15)
    d2 = datetime(2000, 8, 20)
    assert ceil_week(d1) == datetime(2000, 1, 15)
    assert ceil_week(d2) == datetime(2000, 8, 22)


def test_floor_second():
    d1 = datetime(2000, 1, 1, 10, 10, 24, 1000)
    assert floor_second(d1) == datetime(2000, 1, 1, 10, 10, 24)


def test_ceil_second():
    d1 = datetime(2000, 1, 1, 10, 10, 24, 1000)
    assert ceil_second(d1) == datetime(2000, 1, 1, 10, 10, 25)


def test_num_to_datetime():
    limits = num_to_datetime((25552, 27743))
    assert limits[0] == datetime(2039, 12, 17, tzinfo=ZoneInfo("UTC"))
    assert limits[1] == datetime(2045, 12, 16, tzinfo=ZoneInfo("UTC"))

    d = num_to_datetime((27742 + 1.9999999999,))[0]
    assert d.microsecond == 0


def test_datetime_to_num():
    res = datetime_to_num([])
    assert len(res) == 0


def test_timedelta_to_num():
    res = timedelta_to_num([])
    assert len(res) == 0


# Just for test coverage
# TODO: Find a better test
def test_align_limits():
    limits = (2009, 2010)
    align_limits(limits, 1 + 1e-14)
