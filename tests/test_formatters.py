# -*- coding: utf-8 -*-
import warnings
from datetime import datetime, timedelta, tzinfo
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
import pytz

from mizani.labels import (
    label_bytes,
    label_comma,
    label_currency,
    label_custom,
    label_date,
    label_log,
    label_number,
    label_ordinal,
    label_percent,
    label_pvalue,
    label_scientific,
    label_timedelta,
)


def test_label_custom():
    x = [3.987, 2, 42.42]
    labels = ["3.99 USD", "2.00 USD", "42.42 USD"]

    assert label_custom("{:.2f} USD")(x) == labels
    assert label_custom("%.2f USD", style="old")(x) == labels

    label = label_custom("%.2f USD", style="ancient")
    with pytest.raises(ValueError):
        label(x)


def test_label_currency():
    x = [1.232, 99.2334, 4.6, 9, 4500]

    labels = label_currency(prefix="C$", precision=0, big_mark=",")(x)
    assert labels == ["C$1", "C$99", "C$5", "C$9", "C$4,500"]

    labels = label_currency(prefix="C$", precision=0, big_mark=" ")(x)
    assert labels == ["C$1", "C$99", "C$5", "C$9", "C$4 500"]

    labels = label_currency(prefix="$", precision=2)(x)
    assert labels == ["$1.23", "$99.23", "$4.60", "$9.00", "$4500.00"]


def test_label_comma():
    x = [1000, 2, 33000, 400]
    labels = label_comma()(x)
    assert labels == ["1,000", "2", "33,000", "400"]


def test_label_percent():
    label = label_percent()
    # same/nearly same precision values
    assert label([0.12, 0.23, 0.34, 0.45]) == ["12%", "23%", "34%", "45%"]

    assert label([0.12, 0.23, 0.34, 4.5]) == ["12%", "23%", "34%", "450%"]

    # mixed precision values
    assert label([0.12, 0.23, 0.34, 45]) == ["12%", "23%", "34%", "4,500%"]


def test_label_scientific():
    label = label_scientific(2)
    assert label([0.12, 0.2376, 0.34, 45]) == [
        "1.20e-01",
        "2.38e-01",
        "3.40e-01",
        "4.50e+01",
    ]

    assert label([0.12, 230, 0.34 * 10**5, 0.4]) == [
        "1.2e-01",
        "2.3e+02",
        "3.4e+04",
        "4.0e-01",
    ]


def test_label_number():
    label = label_number()
    assert label([5, 10, 100, 150]) == ["5", "10", "100", "150"]
    assert label([5, 10, 100, 150e8]) == [
        "5",
        "10",
        "100",
        "15,000,000,000",
    ]

    label = label_number(big_mark="#")
    assert label([1000, 10000]) == ["1#000", "10#000"]

    label = label_number(precision=2, decimal_mark=",")
    assert label([98.23, 34.67]) == ["98,23", "34,67"]

    label = label_number(style_negative="hyphen")
    assert label([-1, 0, 1]) == ["\u22121", "0", "1"]

    with pytest.raises(ValueError):
        label_number(accuracy=0.01, precision=2)


def test_label_log():
    label = label_log()
    assert label([0.001, 0.1, 100]) == ["0.001", "0.1", "100"]
    assert label([0.001, 0.1, 10000]) == ["1e-3", "1e-1", "1e4"]
    assert label([35, 60]) == ["35", "60"]
    assert label([34.99999999999, 60.0000000001]) == ["35", "60"]
    assert label([300.0000000000014, 499.999999999999]) == [
        "300",
        "500",
    ]
    assert label([1, 35, 60, 1000]) == ["1", "35", "60", "1000"]
    assert label([1, 35, 60, 10000]) == ["1", "35", "60", "10000"]
    assert label([3.000000000000001e-05]) == ["3e-5"]
    assert label([1, 1e4]) == ["1", "1e4"]
    assert label([1, 35, 60, 1e6]) == ["1", "4e1", "6e1", "1e6"]

    label = label_log(base=2)
    assert label([1, 2, 4, 8]) == ["2^0", "2^1", "2^2", "2^3"]
    assert label([0b1, 0b10, 0b11]) == ["2^0", "2^1", "2^1.585"]

    label = label_log(base=8)
    assert label([1, 4, 8, 64]) == ["8^0", "8^0.667", "8^1", "8^2"]

    label = label_log(base=5)
    assert label([1, 5, 25, 125]) == ["5^0", "5^1", "5^2", "5^3"]

    label = label_log(base=np.e)
    assert label([1, np.pi, np.e**2, np.e**3]) == [
        "e^0",
        "e^1.145",
        "e^2",
        "e^3",
    ]

    # mathtex
    label = label_log(mathtex=True)
    assert label([0.001, 0.1, 10000]) == [
        "$10^{-3}$",
        "$10^{-1}$",
        "$10^{4}$",
    ]
    assert label([35, 60]) == ["35", "60"]
    assert label([1, 10000]) == ["$10^{0}$", "$10^{4}$"]

    label = label_log(base=8, mathtex=True)
    assert label([1, 4, 64]) == ["$8^{0}$", "$8^{0.667}$", "$8^{2}$"]


def test_label_date():
    x = pd.date_range("1/1/2010", periods=4, freq="4AS")
    result = label_date("%Y")(x)
    assert result == ["2010", "2014", "2018", "2022"]

    x = [datetime(year=2005 + i, month=i, day=i) for i in range(1, 5)]
    result = label_date("%Y:%m:%d")(x)
    assert result == ["2006:01:01", "2007:02:02", "2008:03:03", "2009:04:04"]

    # Timezone with Daylight time
    NY = ZoneInfo("America/New_York")
    x = [datetime(2023, 10, 1, tzinfo=NY), datetime(2023, 11, 1, tzinfo=NY)]
    result = label_date()(x)
    assert result == ["2023-10-01", "2023-11-01"]

    # Same as above, but different tz library
    NY = pytz.timezone("America/New_York")
    x = [datetime(2023, 10, 1, tzinfo=NY), datetime(2023, 11, 1, tzinfo=NY)]
    result = label_date()(x)
    assert result == ["2023-10-01", "2023-11-01"]

    # Unknown Timezone
    class myTzInfo(tzinfo):
        def __str__(self):
            return "None"

    TZ = myTzInfo()
    x = [datetime(2023, 10, 1, tzinfo=TZ), datetime(2023, 11, 1, tzinfo=TZ)]
    with pytest.raises(NotImplementedError, match=r"^a tzinfo subclass"):
        label_date()(x)


def test_label_timedelta():
    x = [timedelta(days=7 * i) for i in range(5)]
    labels = label_timedelta()(x)
    assert labels == ["0 weeks", "1 week", "2 weeks", "3 weeks", "4 weeks"]

    x = [pd.Timedelta(seconds=600 * i) for i in range(5)]
    labels = label_timedelta()(x)
    assert labels == [
        "0 min",
        "10 min",
        "20 min",
        "30 min",
        "40 min",
    ]

    # specific units
    labels = label_timedelta(units="h")(x)
    assert labels == [
        "0.00 h",
        "0.17 h",
        "0.33 h",
        "0.50 h",
        "0.67 h",
    ]

    # usetex
    x = [timedelta(microseconds=7 * i) for i in range(5)]
    labels = label_timedelta(units="us", space=False, usetex=True)(x)
    assert labels == [
        "0$\\mu s$",
        "7$\\mu s$",
        "14$\\mu s$",
        "21$\\mu s$",
        "28$\\mu s$",
    ]


def test_label_pvalue():
    x = [0.90, 0.15, 0.015, 0.009, 0.0005]
    labels = label_pvalue()(x)
    assert labels == ["0.9", "0.15", "0.015", "0.009", "<0.001"]

    labels = label_pvalue(add_p=True)(x)
    assert labels == ["p=0.9", "p=0.15", "p=0.015", "p=0.009", "p<0.001"]

    with warnings.catch_warnings(record=True) as record:
        x = [0.90, 0.15, np.nan, 0.015, 0.009, 0.0005]
        labels = label_pvalue()(x)
        assert labels == ["0.9", "0.15", "nan", "0.015", "0.009", "<0.001"]
        assert not record, "Issued an unexpected warning"

    # NaN is handled without any warning
    assert len(record) == 0


def test_label_ordinal():
    labels = label_ordinal()(range(110, 115))
    assert labels == ["110th", "111th", "112th", "113th", "114th"]

    labels = label_ordinal()(range(120, 125))
    assert labels == ["120th", "121st", "122nd", "123rd", "124th"]

    labels = label_ordinal(big_mark=",")(range(1200, 1205))
    assert labels == ["1,200th", "1,201st", "1,202nd", "1,203rd", "1,204th"]

    labels = label_ordinal(big_mark=".")(range(1200, 1205))
    assert labels == ["1.200th", "1.201st", "1.202nd", "1.203rd", "1.204th"]


def test_label_bytes():
    x = [1000, 1000000, 4e5]
    labels = label_bytes(symbol="MiB")(x)
    assert labels == ["0 MiB", "1 MiB", "0 MiB"]

    labels = label_bytes(symbol="MiB", fmt="{:.2f} ")(x)
    assert labels == ["0.00 MiB", "0.95 MiB", "0.38 MiB"]

    with pytest.raises(ValueError):
        label_bytes(symbol="Bad")(x)


def test_empty_breaks():
    x = []
    assert label_custom()(x) == []
    assert label_comma()(x) == []
    assert label_currency()(x) == []
    assert label_percent()(x) == []
    assert label_scientific()(x) == []
    assert label_date()(x) == []
    assert label_number()(x) == []
    assert label_log()(x) == []
    assert label_timedelta()(x) == []
    assert label_pvalue()(x) == []
    assert label_ordinal()(x) == []
    assert label_bytes()(x) == []
