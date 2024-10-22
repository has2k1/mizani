from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from mizani._core.dates import _from_ordinalf
from mizani.breaks import (
    breaks_date,
    breaks_extended,
    breaks_log,
    breaks_timedelta,
    minor_breaks,
    minor_breaks_trans,
)
from mizani.transforms import log_trans, trans


def test_log_breaks():
    x = [2, 20, 2000]
    limits = min(x), max(x)
    breaks = breaks_log()(limits)
    npt.assert_array_equal(breaks, [1, 10, 100, 1000, 10000])

    breaks = breaks_log(3)(limits)
    npt.assert_array_equal(breaks, [1, 100, 10000])

    breaks = breaks_log()((10000, 10000))
    npt.assert_array_equal(breaks, [10000])

    breaks = breaks_log()((float("-inf"), float("inf")))
    assert len(breaks) == 0

    # When the limits are in the same order of magnitude
    breaks = breaks_log()([35, 60])
    assert len(breaks) > 0
    assert all(1 < b < 100 for b in breaks)

    breaks = breaks_log()([200, 800])
    npt.assert_array_equal(breaks, [100, 200, 300, 500, 1000])

    breaks = breaks_log()((1664, 14008))
    npt.assert_array_equal(breaks, [1000, 3000, 5000, 10000, 30000])

    breaks = breaks_log()([407, 3430])
    npt.assert_array_equal(breaks, [300, 500, 1000, 3000, 5000])

    breaks = breaks_log()([1761, 8557])
    npt.assert_array_equal(breaks, [1000, 2000, 3000, 5000, 10000])

    # log_breaks -> _log_sub_breaks -> extended_breaks
    breaks = breaks_log(13)([1, 10])
    npt.assert_array_almost_equal(breaks, np.arange(0, 11))

    # No overflow effects
    breaks = breaks_log(n=6)([1e25, 1e30])
    npt.assert_array_almost_equal(breaks, [1e25, 1e26, 1e27, 1e28, 1e29, 1e30])

    # No overflow effects in _log_sub_breaks
    breaks = breaks_log()([2e19, 8e19])
    npt.assert_array_almost_equal(
        breaks, [1.0e19, 2.0e19, 3.0e19, 5.0e19, 1.0e20]
    )

    # _log_sub_breaks for base != 10
    breaks = breaks_log(n=5, base=60)([2e5, 8e5])
    npt.assert_array_almost_equal(
        breaks, [129600, 216000, 432000, 648000, 1080000]
    )

    breaks = breaks_log(n=5, base=2)([20, 80])
    npt.assert_array_almost_equal(breaks, [16, 32, 64, 128])

    # bases & negative breaks
    breaks = breaks_log(base=2)([0.9, 2.9])
    npt.assert_array_almost_equal(breaks, [0.5, 1.0, 2.0, 4.0])


def test_minor_breaks():
    # equidistant breaks
    major = [1, 2, 3, 4]
    limits = [0, 5]
    breaks = minor_breaks()(major, limits)
    npt.assert_array_equal(breaks, [0.5, 1.5, 2.5, 3.5, 4.5])
    minor = minor_breaks(3)(major, [2, 3])
    npt.assert_array_equal(minor, [2.25, 2.5, 2.75])

    # More than 1 minor breaks
    breaks = minor_breaks()(major, limits, 3)
    npt.assert_array_equal(
        breaks,
        [
            0.25,
            0.5,
            0.75,
            1.25,
            1.5,
            1.75,
            2.25,
            2.5,
            2.75,
            3.25,
            3.5,
            3.75,
            4.25,
            4.5,
            4.75,
        ],
    )

    # non-equidistant breaks
    major = [1, 2, 4, 8]
    limits = [0, 10]
    minor = minor_breaks()(major, limits)
    npt.assert_array_equal(minor, [1.5, 3, 6])

    # single major break
    minor = minor_breaks()([2], limits)
    assert len(minor) == 0


def test_minor_breaks_trans():
    major = [1, 2, 3, 4]
    limits = [0, 5]
    regular_mb = minor_breaks()(major, limits)

    class identity_trans(trans):
        transform_is_linear = True
        transform = staticmethod(lambda x: x)
        inverse = staticmethod(lambda x: x)

    t1 = identity_trans()
    t2 = identity_trans()
    t2.minor_breaks = minor_breaks_trans(t2)
    npt.assert_allclose(regular_mb, t1.minor_breaks(major, limits))
    npt.assert_allclose(regular_mb, t2.minor_breaks(major, limits))

    class square_trans(trans):
        transform = staticmethod(np.square)
        inverse = staticmethod(np.sqrt)

    # Transform the input major breaks and check against
    # the inverse of the output minor breaks
    t = square_trans()
    square_mb = t.minor_breaks(np.square(major), np.square(limits))
    npt.assert_allclose(regular_mb, np.sqrt(square_mb))

    # Test minor_breaks for log scales are 2 less than the base
    base = 10
    breaks = np.arange(1, 3)
    limits = [breaks[0], breaks[-1]]
    t = log_trans(base)
    assert len(t.minor_breaks(breaks, limits)) == base - 2

    base = 5  # Odd base
    breaks = np.arange(1, 3)
    limits = [breaks[0], breaks[-1]]
    t = log_trans(base)
    assert len(t.minor_breaks(breaks, limits)) == base - 2

    t = log_trans()
    major = t.transform([1, 10, 100])
    limits = t.transform([1, 100])
    result = minor_breaks_trans(t)(major, limits, n=4)
    npt.assert_allclose(
        result,
        [
            1.02961942,
            1.5260563,
            1.85629799,
            2.10413415,
            3.33220451,
            3.8286414,
            4.15888308,
            4.40671925,
        ],
    )


def test_breaks_date():
    # cpython
    limits = (datetime(2010, 1, 1), datetime(2026, 1, 1))
    breaks = breaks_date(width="5 Years")
    assert [d.year for d in breaks(limits)] == [2010, 2015, 2020, 2025, 2030]

    breaks = breaks_date(width="10 Years")(limits)
    assert [d.year for d in breaks] == [2010, 2020, 2030]

    with pytest.warns(FutureWarning):
        breaks = breaks_date("10 Years")(limits)
        assert [d.year for d in breaks] == [2010, 2020, 2030]

    # numpy datetime64
    limits = (np.datetime64("1973"), np.datetime64("1997"))
    breaks = breaks_date(width="10 Years")(limits)
    assert [d.year for d in breaks] == [1970, 1980, 1990, 2000]

    # NaT
    limits = np.datetime64("NaT"), datetime(2017, 1, 1)
    breaks = breaks_date(width="10 Years")(limits)
    assert len(breaks) == 0

    # automatic monthly breaks
    limits = (datetime(2020, 1, 1), datetime(2021, 1, 15))
    breaks = breaks_date()(limits)
    assert [dt.month for dt in breaks] == [1, 4, 7, 10, 1]

    # automatic monthly breaks with rounding
    limits = (datetime(2019, 12, 27), datetime(2020, 6, 3))
    breaks = breaks_date()(limits)
    assert [dt.month for dt in breaks] == [1, 3, 5]

    # automatic day breaks
    limits = (datetime(2020, 1, 1), datetime(2020, 1, 15))
    breaks = breaks_date()(limits)
    assert [dt.day for dt in breaks] == [1, 5, 9, 13]

    # automatic second breaks
    limits = (datetime(2020, 1, 1, hour=0), datetime(2020, 1, 1, hour=19))
    breaks = breaks_date()(limits)
    assert [dt.hour for dt in breaks] == [0, 4, 8, 12, 16]

    # automatic minute breaks
    limits = (datetime(2020, 1, 1, minute=0), datetime(2020, 1, 1, minute=50))
    breaks = breaks_date()(limits)
    assert [dt.minute for dt in breaks] == [0, 15, 30, 45]

    # automatic second breaks
    limits = (datetime(2020, 1, 1, second=20), datetime(2020, 1, 1, second=50))
    breaks = breaks_date()(limits)
    assert [dt.second for dt in breaks] == [20, 30, 40, 50]

    # automatic microsecond breaks
    limits = (
        datetime(2020, 1, 1, microsecond=10),
        datetime(2020, 1, 1, microsecond=25),
    )
    breaks = breaks_date()(limits)
    assert [dt.microsecond for dt in breaks] == [5, 10, 15, 20, 25, 30]

    # timezone
    UG = ZoneInfo("Africa/Kampala")
    limits = (datetime(1990, 1, 1, tzinfo=UG), datetime(2022, 1, 1, tzinfo=UG))
    breaks = breaks_date()(limits)
    assert breaks[0].tzinfo == UG

    # Special cases
    limits = (datetime(2039, 12, 17), datetime(2045, 12, 16))
    breaks = breaks_date()(limits)
    assert [dt.year for dt in breaks] == [2038, 2040, 2042, 2044, 2046]

    # error cases
    with pytest.raises(ValueError):
        _from_ordinalf(2.933e6, None)


def test_date_type_breaks():
    limits1 = (date(2020, 1, 1), date(2023, 2, 15))
    limits2 = (datetime(2020, 1, 1), datetime(2023, 2, 15))
    calc_breaks = breaks_date()
    res1 = calc_breaks(limits1)
    res2 = [b.replace(tzinfo=None) for b in calc_breaks(limits2)]
    assert res1 == res2


def _check_width(limits, breaks, td: timedelta):
    padding = abs(breaks[0] - limits[0]) + abs(breaks[-1] - limits[-1])
    assert all(np.diff(breaks) == td)
    assert breaks[0] <= limits[0] and limits[-1] <= breaks[-1]
    assert padding < td


def test_breaks_date_week_auto():
    limits = (datetime(2020, 1, 1), datetime(2020, 2, 15))

    breaks = breaks_date()(limits)
    assert {b.day for b in breaks} == {1, 15}

    breaks = breaks_date(n=8)(limits)
    assert {b.day for b in breaks} == {1, 8, 15, 22}


def test_breaks_date_day_width():
    # days
    # 1. The width should be as specified
    # 2. The breaks should encloses the limits
    # 3. The breaks the padding added around the limits should be less than
    #    the width of the breaks
    limits = (datetime(2020, 1, 1), datetime(2020, 2, 28))
    breaks = breaks_date(width="10 days")(limits)
    _check_width(limits, breaks, timedelta(days=10))

    breaks = breaks_date(width="11 days")(limits)
    _check_width(limits, breaks, timedelta(days=11))

    breaks = breaks_date(width="12 days")(limits)
    _check_width(limits, breaks, timedelta(days=12))

    limits = (
        datetime(2000, 1, 1, hour=2),
        datetime(2000, 1, 1, hour=16, second=13),
    )
    breaks = breaks_date(width="2 hour")(limits)
    _check_width(limits, breaks, timedelta(hours=2))

    breaks = breaks_date(width="100 minutes")(limits)
    _check_width(limits, breaks, timedelta(minutes=100))

    breaks = breaks_date(width="5000 seconds")(limits)
    _check_width(limits, breaks, timedelta(seconds=5000))

    breaks = breaks_date(width="5000 seconds")(limits)
    _check_width(limits, breaks, timedelta(seconds=5000))


def test_breaks_date_week_width():
    # weeks
    # 1. The width should be as specified
    # 2. The breaks should encloses the limits
    # 3. The breaks the padding added around the limits should be less than
    #    the width of the breaks
    limits = (datetime(2020, 1, 1), datetime(2020, 2, 28))
    breaks = breaks_date(width="1 week")(limits)
    _check_width(limits, breaks, timedelta(days=7))

    breaks = breaks_date(width="2 weeks")(limits)
    _check_width(limits, breaks, timedelta(days=14))

    breaks = breaks_date(width="3 weeks")(limits)
    _check_width(limits, breaks, timedelta(days=21))


def test_breaks_date_month_width():
    # months
    # 1. The width is within any combination of sequential months
    # 2. The breaks should enclose the limits
    limits = (datetime(2020, 1, 1), datetime(2021, 2, 28))

    breaks = breaks_date(width="1 month")(limits)
    assert all(28 <= d.days <= 31 for d in np.diff(breaks))
    assert breaks[0] <= limits[0] and limits[-1] <= breaks[-1]

    breaks = breaks_date(width="2 month")(limits)
    assert all(59 <= d.days <= 62 for d in np.diff(breaks))
    assert breaks[0] <= limits[0] and limits[-1] <= breaks[-1]
    #
    breaks = breaks_date(width="3 month")(limits)
    assert all(90 <= d.days <= 92 for d in np.diff(breaks))
    assert breaks[0] <= limits[0] and limits[-1] <= breaks[-1]


def test_breaks_timedelta():
    breaks = breaks_timedelta()

    # cpython
    x = [timedelta(days=i * 365) for i in range(25)]
    limits = min(x), max(x)
    major = breaks(limits)
    years = [val.total_seconds() / (365 * 24 * 60 * 60) for val in major]
    npt.assert_array_equal(years, [0, 5, 10, 15, 20, 25])

    x = [timedelta(microseconds=i) for i in range(25)]
    limits = min(x), max(x)
    major = breaks(limits)
    mseconds = [val.total_seconds() * 10**6 for val in major]
    npt.assert_array_equal(mseconds, [0, 5, 10, 15, 20, 25])

    # pandas
    x = [pd.Timedelta(seconds=i * 60) for i in range(10)]
    limits = min(x), max(x)
    major = breaks(limits)
    minutes = [val.total_seconds() / 60 for val in major]
    npt.assert_allclose(minutes, [0, 2, 4, 6, 8])

    # numpy timedelta64 is not supported
    x = [np.timedelta64(i * 10, "D") for i in range(1, 10)]
    limits = x[0], x[-1]
    with pytest.raises(ValueError):
        breaks(limits)

    # NaT
    limits = pd.NaT, pd.Timedelta(seconds=9 * 60)
    assert len(breaks(limits)) == 0


def test_breaks_extended():
    x = np.arange(100)
    limits = min(x), max(x)
    for n in (5, 7, 10, 13, 31):
        breaks = breaks_extended(n=n)
        assert len(breaks(limits)) <= n + 1

    # Reverse limits
    breaks = breaks_extended(n=7)
    npt.assert_array_equal(breaks((0, 6)), breaks((6, 0)))

    # Infinite limits
    limits = float("-inf"), float("inf")
    breaks = breaks_extended(n=5)
    assert len(breaks(limits)) == 0

    # Zero range discrete
    limits = [1, 1]
    assert len(breaks(limits)) == 1
    assert breaks(limits)[0] == limits[1]

    # Zero range continuous
    limits = [np.pi, np.pi]
    assert len(breaks(limits)) == 1
    assert breaks(limits)[0] == limits[1]
