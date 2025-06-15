from datetime import datetime

from mizani._datetime import rounding


def test_microseconds():
    d = datetime(2025, 6, 10, 10, 10, 24, 1)
    assert rounding.microseconds.floor(d) == d
    assert rounding.microseconds.ceil(d) == d


def test_seconds():
    d1 = datetime(2025, 6, 10, 10, 10, 24, 1000)
    d2 = d1.replace(microsecond=0)

    assert rounding.seconds.floor(d1) == datetime(2025, 6, 10, 10, 10, 24)
    assert rounding.seconds.ceil(d1) == datetime(2025, 6, 10, 10, 10, 25)
    assert rounding.seconds.floor(d2) == d2
    assert rounding.seconds.ceil(d2) == d2


def test_minutes():
    d1 = datetime(2025, 6, 10, 10, 10, 24, 1000)
    d2 = datetime(2025, 6, 10, 10, 10)
    assert rounding.minutes.floor(d1) == datetime(2025, 6, 10, 10, 10)
    assert rounding.minutes.ceil(d1) == datetime(2025, 6, 10, 10, 11)
    assert rounding.minutes.floor(d2) == d2
    assert rounding.minutes.ceil(d2) == d2


def test_hours():
    d = datetime(2025, 6, 10, 10, 10, 24, 1000)
    assert rounding.hours.floor(d) == datetime(2025, 6, 10, 10)
    assert rounding.hours.ceil(d) == datetime(2025, 6, 10, 11)


def test_days():
    d = datetime(2025, 6, 10, 10, 10, 24, 1000)
    assert rounding.days.floor(d) == datetime(2025, 6, 10)
    assert rounding.days.ceil(d) == datetime(2025, 6, 11)


def test_weeks():
    d1 = datetime(2025, 6, 5)
    d2 = datetime(2025, 6, 9)
    d3 = datetime(2025, 6, 9, 0, 0, 1)

    assert rounding.weeks.floor(d1) == datetime(2025, 6, 2)
    assert rounding.weeks.floor(d2) == datetime(2025, 6, 9)
    assert rounding.weeks.floor(d3) == datetime(2025, 6, 9)

    assert rounding.weeks.ceil(d1) == datetime(2025, 6, 9)
    assert rounding.weeks.ceil(d2) == datetime(2025, 6, 9)
    assert rounding.weeks.ceil(d3) == datetime(2025, 6, 16)


def test_mjnths():
    d1 = datetime(2020, 1, 11)
    d2 = datetime(2020, 1, 1)

    assert rounding.months.floor(d1) == datetime(2020, 1, 1)
    assert rounding.months.floor(d2) == datetime(2020, 1, 1)

    assert rounding.months.ceil(d1) == datetime(2020, 2, 1)
    assert rounding.months.ceil(d2) == datetime(2020, 1, 1)


def test_years():
    d1 = datetime(2025, 6, 5)
    d2 = datetime(2025, 1, 1)

    assert rounding.years.floor(d1) == datetime(2025, 1, 1)
    assert rounding.years.floor(d2) == datetime(2025, 1, 1)

    assert rounding.years.ceil(d1) == datetime(2026, 1, 1)
    assert rounding.years.ceil(d2) == datetime(2025, 1, 1)


def test_decades():
    d1 = datetime(2025, 6, 5)
    d2 = datetime(2020, 1, 1, 0, 1)
    d3 = datetime(2020, 1, 1)

    assert rounding.decades.floor(d1) == datetime(2020, 1, 1)
    assert rounding.decades.floor(d2) == datetime(2020, 1, 1)
    assert rounding.decades.floor(d3) == datetime(2020, 1, 1)

    assert rounding.decades.ceil(d1) == datetime(2030, 1, 1)
    assert rounding.decades.ceil(d2) == datetime(2030, 1, 1)
    assert rounding.decades.ceil(d3) == datetime(2020, 1, 1)


def test_centurys():
    d1 = datetime(2025, 1, 1)
    d2 = datetime(2000, 1, 1, 0, 1)
    d3 = datetime(2000, 1, 1)
    assert rounding.centurys.floor(d1) == datetime(2000, 1, 1)
    assert rounding.centurys.floor(d2) == datetime(2000, 1, 1)
    assert rounding.centurys.floor(d3) == datetime(2000, 1, 1)

    assert rounding.centurys.ceil(d1) == datetime(2100, 1, 1)
    assert rounding.centurys.ceil(d2) == datetime(2100, 1, 1)
    assert rounding.centurys.ceil(d3) == datetime(2000, 1, 1)
