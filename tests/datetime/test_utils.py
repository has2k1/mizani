from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pytest
from dateutil.relativedelta import relativedelta

from mizani._datetime.utils import (
    as_relativedelta,
    datetime_to_num,
    dt,
    get_tzinfo,
    num_to_datetime,
    parse_datetime_width,
)


def test_tzinfo():
    tz = ZoneInfo("Africa/Kampala")
    assert get_tzinfo("Africa/Kampala") == tz
    assert get_tzinfo(tz) is tz
    with pytest.raises(TypeError):
        assert get_tzinfo(10)  # type: ignore


def test_datetime_to_num():
    assert len(datetime_to_num([])) == 0


def test_num_to_datetime():
    limits = num_to_datetime((25552, 27743))
    assert limits[0] == datetime(2039, 12, 17, tzinfo=ZoneInfo("UTC"))
    assert limits[1] == datetime(2045, 12, 16, tzinfo=ZoneInfo("UTC"))

    d = num_to_datetime((27742 + 1.9999999999,))[0]
    assert d.microsecond == 0

    limits = (np.datetime64("10000-01-02"), np.datetime64("10000-01-02"))

    with pytest.raises(ValueError):
        num_to_datetime(datetime_to_num(limits))


def test_dt():
    assert dt("2020-02-03 10:11:12") == datetime(2020, 2, 3, 10, 11, 12)


def test_parse_datetime_width():
    assert parse_datetime_width("2 sec") == ("seconds", 2)
    assert parse_datetime_width("2 milliseconds") == ("microseconds", 2000)


def test_as_relativedelta():
    rd = relativedelta(days=2, hours=6)
    assert as_relativedelta(rd) == rd
