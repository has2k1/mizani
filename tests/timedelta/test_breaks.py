from __future__ import annotations

from datetime import timedelta
from functools import wraps

from dateutil.relativedelta import relativedelta

from mizani._timedelta.breaks import by_n as _by_n
from mizani._timedelta.breaks import by_width as _by_width


# timedelta repr uses 3 units; milliseconds, seconds and days.
# This make it hard to look at the breaks and tell if they
# are good values. For better output, we convert the breaks to
# relativedelta type.
@wraps(_by_n)
def by_n(*args, **kwargs):
    return [x + relativedelta() for x in _by_n(*args, **kwargs)]


@wraps(_by_width)
def by_width(*args, **kwargs):
    return [x + relativedelta() for x in _by_width(*args, **kwargs)]


def test_by_n_seconds():
    l1 = (timedelta(seconds=10), timedelta(seconds=25))
    assert by_n(l1) == [
        relativedelta(seconds=+10),
        relativedelta(seconds=+15),
        relativedelta(seconds=+20),
        relativedelta(seconds=+25),
    ]


def test_by_n_minutes():
    l1 = (timedelta(minutes=10), timedelta(minutes=55))
    assert by_n(l1) == [
        relativedelta(minutes=+10),
        relativedelta(minutes=+20),
        relativedelta(minutes=+30),
        relativedelta(minutes=+40),
        relativedelta(minutes=+50),
        relativedelta(hours=+1),
    ]


def test_by_n_hours():
    l1 = (timedelta(hours=1, minutes=2), timedelta(hours=24, minutes=55))
    assert by_n(l1) == [
        relativedelta(),
        relativedelta(hours=+5),
        relativedelta(hours=+10),
        relativedelta(hours=+15),
        relativedelta(hours=+20),
        relativedelta(days=+1, hours=+1),
    ]


def test_by_n_days():
    l1 = (timedelta(days=6, hours=2), timedelta(days=29, hours=4))
    assert by_n(l1) == [
        relativedelta(days=+5),
        relativedelta(days=+10),
        relativedelta(days=+15),
        relativedelta(days=+20),
        relativedelta(days=+25),
        relativedelta(days=+30),
    ]


def test_by_n_weeks():
    l1 = (timedelta(days=50), timedelta(days=345))
    assert by_n(l1) == [
        relativedelta(),
        relativedelta(days=+70),
        relativedelta(days=+140),
        relativedelta(days=+210),
        relativedelta(days=+280),
    ]


def test_by_width_seconds():
    l1 = (timedelta(seconds=10), timedelta(seconds=25))
    assert by_width(l1, "4 seconds") == [
        relativedelta(seconds=+8),
        relativedelta(seconds=+12),
        relativedelta(seconds=+16),
        relativedelta(seconds=+20),
        relativedelta(seconds=+24),
        relativedelta(seconds=+28),
    ]

    assert by_width(l1, "4 seconds", 1) == [
        relativedelta(seconds=+9),
        relativedelta(seconds=+13),
        relativedelta(seconds=+17),
        relativedelta(seconds=+21),
        relativedelta(seconds=+25),
        relativedelta(seconds=+29),
    ]

    assert by_width(l1, "30 seconds") == [
        relativedelta(),
        relativedelta(seconds=+30),
    ]


def test_by_width_minutes():
    l1 = (timedelta(minutes=10), timedelta(minutes=55))
    assert by_width(l1, "5 minutes", "30 seconds") == [
        relativedelta(minutes=+10, seconds=+30),
        relativedelta(minutes=+15, seconds=+30),
        relativedelta(minutes=+20, seconds=+30),
        relativedelta(minutes=+25, seconds=+30),
        relativedelta(minutes=+30, seconds=+30),
        relativedelta(minutes=+35, seconds=+30),
        relativedelta(minutes=+40, seconds=+30),
        relativedelta(minutes=+45, seconds=+30),
        relativedelta(minutes=+50, seconds=+30),
        relativedelta(minutes=+55, seconds=+30),
    ]


def test_by_width_hours():
    l1 = (timedelta(hours=1, minutes=2), timedelta(hours=24, minutes=55))
    assert by_width(l1, "4 hours") == [
        relativedelta(),
        relativedelta(hours=+4),
        relativedelta(hours=+8),
        relativedelta(hours=+12),
        relativedelta(hours=+16),
        relativedelta(hours=+20),
        relativedelta(days=+1),
        relativedelta(days=+1, hours=+4),
    ]


def test_by_width_days():
    l1 = (timedelta(days=6, hours=2), timedelta(days=29, hours=4))
    assert by_width(l1, "6 days") == [
        relativedelta(days=+6),
        relativedelta(days=+12),
        relativedelta(days=+18),
        relativedelta(days=+24),
        relativedelta(days=+30),
    ]
    assert by_width(l1, "6 days", ("6 hours", "30 minutes")) == [
        relativedelta(days=+6, hours=+6, minutes=+30),
        relativedelta(days=+12, hours=+6, minutes=+30),
        relativedelta(days=+18, hours=+6, minutes=+30),
        relativedelta(days=+24, hours=+6, minutes=+30),
        relativedelta(days=+30, hours=+6, minutes=+30),
    ]
    assert by_width(l1, "6 days", ("6 hours", "-30 minutes")) == [
        relativedelta(days=+6, hours=+5, minutes=+30),
        relativedelta(days=+12, hours=+5, minutes=+30),
        relativedelta(days=+18, hours=+5, minutes=+30),
        relativedelta(days=+24, hours=+5, minutes=+30),
        relativedelta(days=+30, hours=+5, minutes=+30),
    ]


def test_by_width_weeks():
    l1 = (timedelta(days=50), timedelta(days=345))
    assert by_width(l1, "10 weeks", 1) == [
        relativedelta(days=+7),
        relativedelta(days=+77),
        relativedelta(days=+147),
        relativedelta(days=+217),
        relativedelta(days=+287),
        relativedelta(days=+357),
    ]
