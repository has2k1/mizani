# -*- coding: utf-8 -*-
from __future__ import division
from datetime import datetime, timedelta

import pandas as pd
import pytest
import pytz

from mizani.formatters import (custom_format, comma_format,
                               currency_format, percent_format,
                               scientific_format, date_format,
                               mpl_format, log_format, timedelta_format)


def test_custom_format():
    x = [3.987, 2, 42.42]
    labels = ['3.99 USD', '2.00 USD', '42.42 USD']
    formatter = custom_format('{:.2f} USD')
    assert formatter(x) == labels

    formatter = custom_format('%.2f USD', style='old')
    assert formatter(x) == labels

    formatter = custom_format('%.2f USD', style='ancient')
    with pytest.raises(ValueError):
        formatter(x)


def test_currency_format():
    x = [1.232, 99.2334, 4.6, 9, 4500]
    formatter = currency_format('C$', digits=0, big_mark=',')
    result = formatter(x)
    assert result == ['C$1', 'C$99', 'C$5', 'C$9', 'C$4,500']

    formatter = currency_format('C$', digits=0, big_mark=' ')
    result = formatter(x)
    assert result == ['C$1', 'C$99', 'C$5', 'C$9', 'C$4 500']

    formatter = currency_format('$', digits=2)
    result = formatter(x)
    assert result == ['$1.23', '$99.23', '$4.60', '$9.00', '$4500.00']


def test_comma_format():
    x = [1000, 2, 33000, 400]
    result = comma_format()(x)
    assert result == ['1,000', '2', '33,000', '400']


def test_percent_format():
    formatter = percent_format()
    # same/nearly same precision values
    assert formatter([.12, .23, .34, .45]) == \
        ['12%', '23%', '34%', '45%']

    assert formatter([.12, .23, .34, 4.5]) == \
        ['12%', '23%', '34%', '450%']

    # mixed precision values
    assert formatter([.12, .23, .34, 45]) == \
        ['10%', '20%', '30%', '4500%']


def test_scientific():
    formatter = scientific_format(2)
    assert formatter([.12, .2376, .34, 45]) == \
        ['1.20e-01', '2.38e-01', '3.40e-01', '4.50e+01']

    assert formatter([.12, 230, .34*10**5, .4]) == \
        ['1.2e-01', '2.3e+02', '3.4e+04', '4.0e-01']


def test_mpl_format():
    formatter = mpl_format()
    assert formatter([5, 10, 100, 150]) == ['5', '10', '100', '150']

    # trigger the order of magnitude correction
    assert formatter([5, 10, 100, 150e8]) == ['0', '0', '0', '1.5e10']


def test_log_format():
    formatter = log_format()

    assert formatter([0.001, 0.1, 100]) == ['0.001', '0.1', '100']
    assert formatter([0.001, 0.1, 1000]) == ['1e-3', '1e-1', '1e3']
    assert formatter([35, 60]) == ['35', '60']
    assert formatter([34.99999999999, 60.0000000001]) == ['35', '60']
    assert formatter([1, 35, 60, 1000]) == ['1', '35', '60', '1000']
    assert formatter([1, 35, 60, 10000]) == ['1', '', '', '1e4']

    formatter = log_format()
    assert formatter([1, 35, 60, 10000]) == ['1', '', '', '1e4']

    formatter = log_format(base=2)
    assert formatter([1, 10, 11, 1011]) == ['1', '10', '11', '1011']


def test_date_format():
    x = pd.date_range('1/1/2010', periods=4, freq='4AS')
    result = date_format('%Y')(x)
    assert result == ['2010', '2014', '2018', '2022']

    x = [datetime(year=2005+i, month=i, day=i) for i in range(1, 5)]
    result = date_format('%Y:%m:%d')(x)
    assert result == \
        ['2006:01:01', '2007:02:02', '2008:03:03', '2009:04:04']

    # Different timezones
    pct = pytz.timezone('US/Pacific')
    ug = pytz.timezone('Africa/Kampala')
    x = [datetime(2010, 1, 1, tzinfo=ug),
         datetime(2010, 1, 1, tzinfo=pct)]
    with pytest.warns(UserWarning):
        date_format()(x)


def test_timedelta_format():
    x = [timedelta(days=7*i) for i in range(5)]
    labels = timedelta_format()(x)
    assert labels == ['0', '1 week', '2 weeks', '3 weeks', '4 weeks']

    x = [pd.Timedelta(seconds=600*i) for i in range(5)]
    labels = timedelta_format()(x)
    assert labels == \
        ['0', '10 minutes', '20 minutes', '30 minutes', '40 minutes']

    # specific units
    labels = timedelta_format(units='h')(x)
    assert labels == \
        ['0', '0.1667 hours', '0.3333 hours', '0.5000 hours',
         '0.6667 hours']
    # usetex
    x = [timedelta(microseconds=7*i) for i in range(5)]
    labels = timedelta_format(units='us', usetex=True)(x)
    assert labels == \
        ['0', '7$\\mu s$', '14$\\mu s$', '21$\\mu s$', '28$\\mu s$']


def test_empty_breaks():
    x = []
    assert custom_format()(x) == []
    assert comma_format()(x) == []
    assert currency_format()(x) == []
    assert percent_format()(x) == []
    assert scientific_format()(x) == []
    assert date_format()(x) == []
    assert mpl_format()(x) == []
    assert log_format()(x) == []
    assert timedelta_format()(x) == []
