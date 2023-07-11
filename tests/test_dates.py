from datetime import datetime

import pytest
from zoneinfo import ZoneInfo

from mizani._core.dates import (
    _align_limits,
    _ceil_mid_year,
    _floor_mid_year,
    datetime_to_num,
    get_tzinfo,
    num_to_datetime,
)


def test_tzinfo():
    tz = ZoneInfo("Africa/Kampala")
    assert get_tzinfo("Africa/Kampala") == tz
    assert get_tzinfo(tz) is tz
    with pytest.raises(TypeError):
        assert get_tzinfo(10)  # type: ignore


def test_floor_mid_year():
    d = datetime(2022, 3, 1)
    assert _floor_mid_year(d) == datetime(2022, 1, 1)


def test_ceil_mid_year():
    d1 = datetime(2022, 1, 1)
    d2 = datetime(2022, 1, 2)
    d3 = datetime(2022, 8, 2)
    assert _ceil_mid_year(d1) == datetime(2022, 1, 1)
    assert _ceil_mid_year(d2) == datetime(2022, 7, 1)
    assert _ceil_mid_year(d3) == datetime(2023, 1, 1)


def test_num_to_datetime():
    limits = num_to_datetime((25552, 27743))
    assert limits[0] == datetime(2039, 12, 17, tzinfo=ZoneInfo("UTC"))
    assert limits[1] == datetime(2045, 12, 16, tzinfo=ZoneInfo("UTC"))

    d = num_to_datetime((27742 + 1.9999999999,))[0]
    assert d.microsecond == 0


def test_datetime_to_num():
    x = []
    res = datetime_to_num([])
    assert len(res) == 0


# Just for test coverage
# TODO: Find a better test
def test_align_limits():
    limits = (2009, 2010)
    _align_limits(limits, 1 + 1e-14)
