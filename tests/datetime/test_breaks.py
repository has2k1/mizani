import sys
from datetime import datetime, timedelta

import pytest

from mizani._datetime.breaks import by_n, by_width
from mizani._datetime.utils import dt

LT_PY311 = sys.version_info < (3, 11)


def test_by_n_microseconds():
    l1 = dt(("2025-01-01 01:10:30:000250", "2025-01-01 01:10:30:000600"))
    assert by_n(l1) == [
        datetime(2025, 1, 1, 1, 10, 30, 200),
        datetime(2025, 1, 1, 1, 10, 30, 300),
        datetime(2025, 1, 1, 1, 10, 30, 400),
        datetime(2025, 1, 1, 1, 10, 30, 500),
        datetime(2025, 1, 1, 1, 10, 30, 600),
    ]
    assert by_n(l1, 10) == [
        datetime(2025, 1, 1, 1, 10, 30, 250),
        datetime(2025, 1, 1, 1, 10, 30, 300),
        datetime(2025, 1, 1, 1, 10, 30, 350),
        datetime(2025, 1, 1, 1, 10, 30, 400),
        datetime(2025, 1, 1, 1, 10, 30, 450),
        datetime(2025, 1, 1, 1, 10, 30, 500),
        datetime(2025, 1, 1, 1, 10, 30, 550),
        datetime(2025, 1, 1, 1, 10, 30, 600),
    ]


def test_by_n_seconds():
    l1 = dt(("2025-01-01 01:10:30", "2025-01-01 01:11:30"))
    l2 = dt(("2025-01-01 01:10:30", "2025-01-01 01:11:31"))
    assert by_n(l1) == [
        datetime(2025, 1, 1, 1, 10, 30),
        datetime(2025, 1, 1, 1, 10, 40),
        datetime(2025, 1, 1, 1, 10, 50),
        datetime(2025, 1, 1, 1, 11),
        datetime(2025, 1, 1, 1, 11, 10),
        datetime(2025, 1, 1, 1, 11, 20),
        datetime(2025, 1, 1, 1, 11, 30),
    ]
    assert by_n(l2) == [
        datetime(2025, 1, 1, 1, 10, 30),
        datetime(2025, 1, 1, 1, 10, 45),
        datetime(2025, 1, 1, 1, 11),
        datetime(2025, 1, 1, 1, 11, 15),
        datetime(2025, 1, 1, 1, 11, 30),
        datetime(2025, 1, 1, 1, 11, 45),
    ]


def test_by_n_minutes():
    l1 = dt(("2025-01-01 01:10:30", "2025-01-01 01:21:45"))
    l2 = dt(("2025-01-01 01:10:30", "2025-01-01 04:00:30"))
    assert by_n(l1) == [
        datetime(2025, 1, 1, 1, 10),
        datetime(2025, 1, 1, 1, 12),
        datetime(2025, 1, 1, 1, 14),
        datetime(2025, 1, 1, 1, 16),
        datetime(2025, 1, 1, 1, 18),
        datetime(2025, 1, 1, 1, 20),
        datetime(2025, 1, 1, 1, 22),
    ]
    assert by_n(l2) == [
        datetime(2025, 1, 1, 1, 0),
        datetime(2025, 1, 1, 1, 30),
        datetime(2025, 1, 1, 2, 0),
        datetime(2025, 1, 1, 2, 30),
        datetime(2025, 1, 1, 3, 0),
        datetime(2025, 1, 1, 3, 30),
        datetime(2025, 1, 1, 4, 0),
        datetime(2025, 1, 1, 4, 30),
    ]
    assert by_n(l1, n=3) == [
        datetime(2025, 1, 1, 1, 10),
        datetime(2025, 1, 1, 1, 15),
        datetime(2025, 1, 1, 1, 20),
        datetime(2025, 1, 1, 1, 25),
    ]


def test_by_n_hours():
    l1 = dt(("2025-01-01 01:10:30", "2025-01-01 05:10:30"))
    l2 = dt(("2025-01-01 01:10:30", "2025-01-02 05:10:30"))
    l3 = dt(("2025-01-01 01:10:30", "2025-01-03 05:10:30"))
    assert by_n(l1) == [
        datetime(2025, 1, 1, 1, 0),
        datetime(2025, 1, 1, 2, 0),
        datetime(2025, 1, 1, 3, 0),
        datetime(2025, 1, 1, 4, 0),
        datetime(2025, 1, 1, 5, 0),
        datetime(2025, 1, 1, 6, 0),
    ]
    assert by_n(l2) == [
        datetime(2025, 1, 1, 0, 0),
        datetime(2025, 1, 1, 6, 0),
        datetime(2025, 1, 1, 12, 0),
        datetime(2025, 1, 1, 18, 0),
        datetime(2025, 1, 2, 0, 0),
        datetime(2025, 1, 2, 6, 0),
    ]
    assert by_n(l3) == [
        datetime(2025, 1, 1, 0, 0),
        datetime(2025, 1, 1, 12, 0),
        datetime(2025, 1, 2, 0, 0),
        datetime(2025, 1, 2, 12, 0),
        datetime(2025, 1, 3, 0, 0),
        datetime(2025, 1, 3, 12, 0),
    ]


def test_by_n_days():
    l1 = dt(("2025-01-01 01:10:30", "2025-01-06 05:10:30"))
    l2 = dt(("2025-01-01 01:10:30", "2025-01-12 05:10:30"))
    assert by_n(l1) == [
        datetime(2025, 1, 1, 0, 0),
        datetime(2025, 1, 2, 0, 0),
        datetime(2025, 1, 3, 0, 0),
        datetime(2025, 1, 4, 0, 0),
        datetime(2025, 1, 5, 0, 0),
        datetime(2025, 1, 6, 0, 0),
        datetime(2025, 1, 7, 0, 0),
    ]
    assert by_n(l2) == [
        datetime(2025, 1, 1, 0, 0),
        datetime(2025, 1, 3, 0, 0),
        datetime(2025, 1, 5, 0, 0),
        datetime(2025, 1, 7, 0, 0),
        datetime(2025, 1, 9, 0, 0),
        datetime(2025, 1, 11, 0, 0),
        datetime(2025, 1, 13, 0, 0),
    ]


def test_by_n_weeks():
    l1 = dt(("2025-01-01 01:10:30", "2025-01-26 05:10:30"))
    l2 = dt(("2025-01-01 01:10:30", "2025-02-02 05:10:30"))
    assert by_n(l1) == [
        datetime(2024, 12, 30, 0, 0),
        datetime(2025, 1, 6, 0, 0),
        datetime(2025, 1, 13, 0, 0),
        datetime(2025, 1, 20, 0, 0),
        datetime(2025, 1, 27, 0, 0),
    ]
    assert by_n(l2) == [
        datetime(2024, 12, 30, 0, 0),
        datetime(2025, 1, 6, 0, 0),
        datetime(2025, 1, 13, 0, 0),
        datetime(2025, 1, 20, 0, 0),
        datetime(2025, 1, 27, 0, 0),
        datetime(2025, 2, 3, 0, 0),
    ]


def test_by_n_months():
    l1 = dt(("2025-01-01 01:10:30", "2025-02-22 05:10:30"))
    l2 = dt(("2025-01-01 01:10:30", "2025-08-10 05:10:30"))
    l3 = dt(("2025-01-01 01:10:30", "2026-01-03 05:10:30"))
    l4 = dt(("2025-01-01 01:10:30", "2027-01-03 05:10:30"))
    assert by_n(l1) == [
        datetime(2025, 1, 1, 0, 0),
        datetime(2025, 1, 15, 0, 0),
        datetime(2025, 2, 1, 0, 0),
        datetime(2025, 2, 15, 0, 0),
        datetime(2025, 3, 1, 0, 0),
    ]
    assert by_n(l2) == [
        datetime(2025, 1, 1, 0, 0),
        datetime(2025, 3, 1, 0, 0),
        datetime(2025, 5, 1, 0, 0),
        datetime(2025, 7, 1, 0, 0),
        datetime(2025, 9, 1, 0, 0),
    ]
    assert by_n(l3) == [
        datetime(2025, 1, 1, 0, 0),
        datetime(2025, 4, 1, 0, 0),
        datetime(2025, 7, 1, 0, 0),
        datetime(2025, 10, 1, 0, 0),
        datetime(2026, 1, 1, 0, 0),
        datetime(2026, 4, 1, 0, 0),
    ]
    assert by_n(l4) == [
        datetime(2025, 1, 1, 0, 0),
        datetime(2025, 7, 1, 0, 0),
        datetime(2026, 1, 1, 0, 0),
        datetime(2026, 7, 1, 0, 0),
        datetime(2027, 1, 1, 0, 0),
        datetime(2027, 7, 1, 0, 0),
    ]


def test_by_n_years():
    l1 = dt(("2025-01-01 01:10:30", "2029-01-20 05:10:30"))
    l2 = dt(("2025-01-01 01:10:30", "2045-01-20 05:10:30"))
    assert by_n(l1) == [
        datetime(2025, 1, 1, 0, 0),
        datetime(2026, 1, 1, 0, 0),
        datetime(2027, 1, 1, 0, 0),
        datetime(2028, 1, 1, 0, 0),
        datetime(2029, 1, 1, 0, 0),
        datetime(2030, 1, 1, 0, 0),
    ]
    assert by_n(l2) == [
        datetime(2025, 1, 1, 0, 0),
        datetime(2030, 1, 1, 0, 0),
        datetime(2035, 1, 1, 0, 0),
        datetime(2040, 1, 1, 0, 0),
        datetime(2045, 1, 1, 0, 0),
        datetime(2050, 1, 1, 0, 0),
    ]


def test_by_n_decades():
    l1 = dt(("2025-01-01 01:10:30", "2098-01-20 05:10:30"))
    l2 = dt(("2025-01-01 01:10:30", "2250-01-20 05:10:30"))
    assert by_n(l1) == [
        datetime(2020, 1, 1, 0, 0),
        datetime(2040, 1, 1, 0, 0),
        datetime(2060, 1, 1, 0, 0),
        datetime(2080, 1, 1, 0, 0),
        datetime(2100, 1, 1, 0, 0),
    ]
    assert by_n(l2) == [
        datetime(2000, 1, 1, 0, 0),
        datetime(2050, 1, 1, 0, 0),
        datetime(2100, 1, 1, 0, 0),
        datetime(2150, 1, 1, 0, 0),
        datetime(2200, 1, 1, 0, 0),
        datetime(2250, 1, 1, 0, 0),
        datetime(2300, 1, 1, 0, 0),
    ]


def test_by_n_century():
    l1 = dt(("2025-01-01 01:10:30", "2429-01-20 05:10:30"))
    l2 = dt(("1225-01-01 01:10:30", "2025-01-20 05:10:30"))
    l3 = dt(("0400-01-01 01:10:30", "2025-01-20 05:10:30"))
    assert by_n(l1) == [
        datetime(2000, 1, 1, 0, 0),
        datetime(2100, 1, 1, 0, 0),
        datetime(2200, 1, 1, 0, 0),
        datetime(2300, 1, 1, 0, 0),
        datetime(2400, 1, 1, 0, 0),
        datetime(2500, 1, 1, 0, 0),
    ]
    assert by_n(l2) == [
        datetime(1200, 1, 1, 0, 0),
        datetime(1400, 1, 1, 0, 0),
        datetime(1600, 1, 1, 0, 0),
        datetime(1800, 1, 1, 0, 0),
        datetime(2000, 1, 1, 0, 0),
        datetime(2200, 1, 1, 0, 0),
    ]
    assert by_n(l3) == [
        datetime(400, 1, 1, 0, 0),
        datetime(900, 1, 1, 0, 0),
        datetime(1400, 1, 1, 0, 0),
        datetime(1900, 1, 1, 0, 0),
        datetime(2400, 1, 1, 0, 0),
    ]


def test_by_width_microseconds():
    l1 = dt(("2025-01-01 01:10:30:000250", "2025-01-01 01:10:30:000600"))
    assert by_width(l1, "250 microseconds") == [
        datetime(2025, 1, 1, 1, 10, 30, 250),
        datetime(2025, 1, 1, 1, 10, 30, 500),
        datetime(2025, 1, 1, 1, 10, 30, 750),
    ]
    assert by_width(l1, "250 microseconds", -50) == [
        datetime(2025, 1, 1, 1, 10, 30, 200),
        datetime(2025, 1, 1, 1, 10, 30, 450),
        datetime(2025, 1, 1, 1, 10, 30, 700),
    ]


def test_by_width_seconds():
    l1 = dt(("2025-01-01 01:10:30", "2025-01-01 01:11:30"))
    l2 = dt(("2025-01-01 01:10:30", "2025-01-01 01:11:31"))
    assert by_width(l1, "10 seconds") == [
        datetime(2025, 1, 1, 1, 10, 30),
        datetime(2025, 1, 1, 1, 10, 40),
        datetime(2025, 1, 1, 1, 10, 50),
        datetime(2025, 1, 1, 1, 11),
        datetime(2025, 1, 1, 1, 11, 10),
        datetime(2025, 1, 1, 1, 11, 20),
        datetime(2025, 1, 1, 1, 11, 30),
    ]
    assert by_width(l2, "10 seconds", offset=1) == [
        datetime(2025, 1, 1, 1, 10, 31),
        datetime(2025, 1, 1, 1, 10, 41),
        datetime(2025, 1, 1, 1, 10, 51),
        datetime(2025, 1, 1, 1, 11, 1),
        datetime(2025, 1, 1, 1, 11, 11),
        datetime(2025, 1, 1, 1, 11, 21),
        datetime(2025, 1, 1, 1, 11, 31),
        datetime(2025, 1, 1, 1, 11, 41),
    ]


def test_by_width_minutes():
    l1 = dt(("2025-01-01 01:11:30", "2025-01-01 04:00:30"))
    assert by_width(l1, "45 mins") == [
        datetime(2025, 1, 1, 1, 0),
        datetime(2025, 1, 1, 1, 45),
        datetime(2025, 1, 1, 2, 30),
        datetime(2025, 1, 1, 3, 15),
        datetime(2025, 1, 1, 4, 0),
        datetime(2025, 1, 1, 4, 45),
    ]
    assert by_width(l1, "45 mins", -10) == [
        datetime(2025, 1, 1, 0, 50),
        datetime(2025, 1, 1, 1, 35),
        datetime(2025, 1, 1, 2, 20),
        datetime(2025, 1, 1, 3, 5),
        datetime(2025, 1, 1, 3, 50),
        datetime(2025, 1, 1, 4, 35),
    ]


def test_by_width_hours():
    l1 = dt(("2025-01-01 01:10:30", "2025-01-03 05:10:30"))
    assert by_width(l1, "12 hours") == [
        datetime(2025, 1, 1, 0, 0),
        datetime(2025, 1, 1, 12, 0),
        datetime(2025, 1, 2, 0, 0),
        datetime(2025, 1, 2, 12, 0),
        datetime(2025, 1, 3, 0, 0),
        datetime(2025, 1, 3, 12, 0),
    ]
    assert by_width(l1, "12 hours", "-12 hours") == [
        datetime(2024, 12, 31, 12, 0),
        datetime(2025, 1, 1, 0, 0),
        datetime(2025, 1, 1, 12, 0),
        datetime(2025, 1, 2, 0, 0),
        datetime(2025, 1, 2, 12, 0),
        datetime(2025, 1, 3, 0, 0),
    ]


def test_by_width_days():
    l1 = dt(("2025-01-01 01:10:30", "2025-01-12 05:10:30"))
    assert by_width(l1, "2 days", timedelta(hours=-1, minutes=-30)) == [
        datetime(2024, 12, 31, 22, 30),
        datetime(2025, 1, 2, 22, 30),
        datetime(2025, 1, 4, 22, 30),
        datetime(2025, 1, 6, 22, 30),
        datetime(2025, 1, 8, 22, 30),
        datetime(2025, 1, 10, 22, 30),
        datetime(2025, 1, 12, 22, 30),
    ]
    assert by_width(l1, "2 days", ("-1 hour", "-30 minutes")) == [
        datetime(2024, 12, 31, 22, 30),
        datetime(2025, 1, 2, 22, 30),
        datetime(2025, 1, 4, 22, 30),
        datetime(2025, 1, 6, 22, 30),
        datetime(2025, 1, 8, 22, 30),
        datetime(2025, 1, 10, 22, 30),
        datetime(2025, 1, 12, 22, 30),
    ]


def test_by_width_weeks():
    l1 = dt(("2025-01-01 01:10:30", "2025-02-02 05:10:30"))
    assert by_width(l1, "1 weeks") == [
        datetime(2024, 12, 30, 0, 0),
        datetime(2025, 1, 6, 0, 0),
        datetime(2025, 1, 13, 0, 0),
        datetime(2025, 1, 20, 0, 0),
        datetime(2025, 1, 27, 0, 0),
        datetime(2025, 2, 3, 0, 0),
    ]
    assert by_width(l1, "2 weeks") == [
        datetime(2025, 1, 1, 0, 0),
        datetime(2025, 1, 15, 0, 0),
        datetime(2025, 1, 29, 0, 0),
        datetime(2025, 2, 12, 0, 0),
    ]


def test_by_width_months():
    l1 = dt(("2025-01-01 01:10:30", "2027-01-03 05:10:30"))
    assert by_width(l1, "6 months") == [
        datetime(2025, 1, 1, 0, 0),
        datetime(2025, 7, 1, 0, 0),
        datetime(2026, 1, 1, 0, 0),
        datetime(2026, 7, 1, 0, 0),
        datetime(2027, 1, 1, 0, 0),
        datetime(2027, 7, 1, 0, 0),
    ]
    assert by_width(l1, "6 months", "12 hours") == [
        datetime(2025, 1, 1, 12, 0),
        datetime(2025, 7, 1, 12, 0),
        datetime(2026, 1, 1, 12, 0),
        datetime(2026, 7, 1, 12, 0),
        datetime(2027, 1, 1, 12, 0),
        datetime(2027, 7, 1, 12, 0),
    ]


def test_by_width_years():
    l1 = dt(("2025-01-01 01:10:30", "2045-01-20 05:10:30"))
    assert by_width(l1, "5 years") == [
        datetime(2025, 1, 1, 0, 0),
        datetime(2030, 1, 1, 0, 0),
        datetime(2035, 1, 1, 0, 0),
        datetime(2040, 1, 1, 0, 0),
        datetime(2045, 1, 1, 0, 0),
        datetime(2050, 1, 1, 0, 0),
    ]


def test_by_width_decades():
    l1 = dt(("2025-01-01 01:10:30", "2250-01-20 05:10:30"))
    assert by_width(l1, "2 decades") == [
        datetime(2020, 1, 1, 0, 0),
        datetime(2040, 1, 1, 0, 0),
        datetime(2060, 1, 1, 0, 0),
        datetime(2080, 1, 1, 0, 0),
        datetime(2100, 1, 1, 0, 0),
        datetime(2120, 1, 1, 0, 0),
        datetime(2140, 1, 1, 0, 0),
        datetime(2160, 1, 1, 0, 0),
        datetime(2180, 1, 1, 0, 0),
        datetime(2200, 1, 1, 0, 0),
        datetime(2220, 1, 1, 0, 0),
        datetime(2240, 1, 1, 0, 0),
        datetime(2260, 1, 1, 0, 0),
    ]
    assert by_width(l1, "4 decades", 5) == [
        datetime(2005, 1, 1, 0, 0),
        datetime(2045, 1, 1, 0, 0),
        datetime(2085, 1, 1, 0, 0),
        datetime(2125, 1, 1, 0, 0),
        datetime(2165, 1, 1, 0, 0),
        datetime(2205, 1, 1, 0, 0),
        datetime(2245, 1, 1, 0, 0),
        datetime(2285, 1, 1, 0, 0),
    ]


def test_by_width_century():
    l1 = dt(("0400-01-01 01:10:30", "2025-01-20 05:10:30"))
    assert by_width(l1, "2 centuries") == [
        datetime(400, 1, 1, 0, 0),
        datetime(600, 1, 1, 0, 0),
        datetime(800, 1, 1, 0, 0),
        datetime(1000, 1, 1, 0, 0),
        datetime(1200, 1, 1, 0, 0),
        datetime(1400, 1, 1, 0, 0),
        datetime(1600, 1, 1, 0, 0),
        datetime(1800, 1, 1, 0, 0),
        datetime(2000, 1, 1, 0, 0),
        datetime(2200, 1, 1, 0, 0),
    ]


@pytest.mark.skipif(
    LT_PY311, reason="python < 3.11 does not support some isoformats"
)
def test_breaks_timezone():
    l1 = dt(("2025-01-01 01:10:30+03", "2025-01-03 05:10:30+03"))
    assert all(b.tzinfo is not None for b in by_n(l1))
