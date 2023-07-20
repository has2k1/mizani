from datetime import datetime, timedelta

from mizani._core.date_utils import expand_datetime_limits, shift_limits_down


def test_shift_limits_down():
    lo = (1973, 1998)
    lc = (1973, 2023)
    assert shift_limits_down(lc, lo, 10) == (1970, 2020)

    lo = (1973, 2021)
    lc = (1973, 2023)
    assert shift_limits_down(lc, lo, 10) == lc


def test_expand_datetime_limits():
    limits = (
        datetime(2000, 1, 1, microsecond=100),
        datetime(2000, 1, 1, microsecond=1300),
    )
    assert expand_datetime_limits(limits, 200, "microsecond") == limits
