from datetime import timedelta

import pytest
from dateutil.relativedelta import relativedelta

from mizani._timedelta.utils import (
    as_timedelta,
    parse_timedelta_width,
    timedelta_to_num,
)


def test_timedelta_to_num():
    res = timedelta_to_num([])
    assert len(res) == 0


def test_parse_timedelta_width():
    assert parse_timedelta_width("2 sec") == ("seconds", 2)
    assert parse_timedelta_width("-5 min") == ("minutes", -5)


def test_as_timedelta():
    td = timedelta(days=2, hours=6)
    rd = relativedelta(days=2, hours=6)
    assert as_timedelta(td) == td
    assert as_timedelta(rd) == td  # pyright: ignore[reportArgumentType]

    with pytest.raises(ValueError):
        rd = relativedelta(months=11, days=2, hours=6)
        as_timedelta(rd)  # pyright: ignore[reportArgumentType]
