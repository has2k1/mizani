from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from mizani.utils import (
    get_categories,
    get_null_value,
    get_timezone,
    match,
    min_max,
    precision,
    round_any,
    same_log10_order_of_magnitude,
)


def test_round_any():
    x = 4.632
    assert round_any(x, 1) == 5
    assert round_any(x, 2) == 4
    assert round_any(x, 3) == 6
    assert round_any(x, 4) == 4
    assert round_any(x, 5) == 5
    assert round_any(x, 1.5) == 4.5


def test_min_max():
    x = [1, 2, 3, 4, 5]
    _min, _max = min_max(x)
    assert _min == 1
    assert _max == 5

    x = [1, float("-inf"), 3, 4, 5]
    _min, _max = min_max(x)
    assert _min == 1
    assert _max == 5

    _min, _max = min_max(x, finite=False)
    assert _min == float("-inf")
    assert _max == 5

    x = [1, 2, float("nan"), 4, 5]
    _min, _max = min_max(x, na_rm=True)
    assert _min == 1
    assert _max == 5

    x = [1, 2, float("nan"), 4, 5, float("inf")]
    _min, _max = min_max(x, na_rm=True, finite=False)
    assert _min == 1
    assert _max == float("inf")

    _min, _max = min_max(x)
    assert str(_min) == "nan"
    assert str(_max) == "nan"

    x = [float("nan"), float("nan"), float("nan")]
    _min, _max = min_max(x, na_rm=True)
    assert _min == float("-inf")
    assert _max == float("inf")


def test_match():
    v1 = [0, 1, 2, 3, 4, 5]
    v2 = [5, 4, 3, 2, 1, 0]
    result = match(v1, v2)
    assert result == v2

    # Positions of the first match
    result = match(v1, v2 + v2)
    assert result == v2

    result = match(v1, v2, incomparables=[1, 2])
    assert result == [5, -1, -1, 2, 1, 0]

    result = match(v1, v2, start=1)
    assert result == [6, 5, 4, 3, 2, 1]

    v2 = [5, 99, 3, 2, 1, 0]
    result = match(v1, v2)
    assert result == [5, 4, 3, 2, -1, 0]


def test_precision():
    assert precision(0.0037) == 1
    assert precision([0.0037, 0.0045]) == 0.0001
    assert precision([0.5, 0.4]) == 0.1
    assert precision([5, 9]) == 1
    assert precision([24, 84]) == 1
    assert precision([290, 784]) == 1
    assert precision([0.0037, 0.5, 9, 24, 784]) == 0.1


def test_same_log10_order_of_magnitude():
    # Default delta
    assert same_log10_order_of_magnitude((2, 8))
    assert same_log10_order_of_magnitude((35, 80.8))
    assert same_log10_order_of_magnitude((232.3, 730))

    assert not same_log10_order_of_magnitude((1, 18))
    assert not same_log10_order_of_magnitude((35, 800))
    assert not same_log10_order_of_magnitude((32, 730))

    assert not same_log10_order_of_magnitude((1, 9.9))
    assert not same_log10_order_of_magnitude((35, 91))
    assert not same_log10_order_of_magnitude((232.3, 950))

    # delta = 0
    assert same_log10_order_of_magnitude((1, 9.9), delta=0)
    assert same_log10_order_of_magnitude((35, 91), delta=0)
    assert same_log10_order_of_magnitude((232.3, 950), delta=0)


def test_get_categories():
    lst = list("abcd")
    s = pd.Series(lst)
    c = pd.Categorical(lst)
    sc = pd.Series(c)

    categories = pd.Index(lst)
    assert categories.equals(get_categories(c))
    assert categories.equals(get_categories(sc))

    with pytest.raises(TypeError):
        assert categories.equals(get_categories(lst))

    with pytest.raises(TypeError):
        assert categories.equals(get_categories(s))


def test_get_timezone():
    UTC = ZoneInfo("UTC")
    UG = ZoneInfo("Africa/Kampala")

    x = [date(2022, 1, 1), date(2022, 12, 1)]
    assert get_timezone(x) is None

    x = [datetime(2022, 1, 1, tzinfo=UTC), datetime(2022, 12, 1, tzinfo=UG)]
    with pytest.warns(UserWarning, match="^Dates"):
        get_timezone(x)


def test_get_null_value():
    x = [datetime(2022, 3, 24)]
    assert get_null_value(x) is None
