from __future__ import division
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import numpy.testing as npt
import pytest

from ..breaks import (mpl_breaks, log_breaks, minor_breaks,
                      trans_minor_breaks, date_breaks,
                      timedelta_breaks)
from ..transforms import trans


def test_mpl_breaks():
    x = np.arange(100)
    limits = min(x), max(x)
    for nbins in (5, 7, 10, 13, 31):
        breaks = mpl_breaks(nbins=nbins)
        assert len(breaks(limits)) <= nbins+1


def test_log_breaks():
    x = [2, 20, 2000]
    limits = min(x), max(x)
    breaks = log_breaks()(limits)
    npt.assert_array_equal(breaks, [1, 10, 100, 1000, 10000])

    breaks = log_breaks(3)(limits)
    npt.assert_array_equal(breaks, [1, 100, 10000])

    breaks = log_breaks()((10000, 10000))
    npt.assert_array_equal(breaks, [10000])


def test_minor_breaks():
    # equidistant breaks
    major = [1, 2, 3, 4]
    limits = [0, 5]
    breaks = minor_breaks()(major, limits)
    npt.assert_array_equal(breaks, [.5, 1.5, 2.5, 3.5, 4.5])
    minor = minor_breaks(3)(major, [2, 3])
    npt.assert_array_equal(minor, [2.25, 2.5, 2.75])

    # non-equidistant breaks
    major = [1, 2, 4, 8]
    limits = [0, 10]
    minor = minor_breaks()(major, limits)
    npt.assert_array_equal(minor, [1.5, 3, 6])

    # single major break
    minor = minor_breaks()([2], limits)
    assert len(minor) == 0


def test_trans_minor_breaks():
    class identity_trans(trans):
        minor_breaks = trans_minor_breaks()

    class square_trans(trans):
        transform = staticmethod(np.square)
        inverse = staticmethod(np.sqrt)
        minor_breaks = trans_minor_breaks()

    class weird_trans(trans):
        dataspace_is_numerical = False
        minor_breaks = trans_minor_breaks()

    major = [1, 2, 3, 4]
    limits = [0, 5]
    regular_minors = trans.minor_breaks(major, limits)
    npt.assert_allclose(
        regular_minors,
        identity_trans.minor_breaks(major, limits))

    # Transform the input major breaks and check against
    # the inverse of the output minor breaks
    squared_input_minors = square_trans.minor_breaks(
                np.square(major), np.square(limits))
    npt.assert_allclose(regular_minors,
                        np.sqrt(squared_input_minors))

    t = weird_trans()
    with pytest.raises(TypeError):
        t.minor_breaks(major)


def test_date_breaks():
    # cpython
    x = [datetime(year, 1, 1) for year in [2010, 2026, 2015]]
    limits = min(x), max(x)

    breaks = date_breaks('5 Years')
    years = [d.year for d in breaks(limits)]
    npt.assert_array_equal(
        years, [2010, 2015, 2020, 2025, 2030])

    breaks = date_breaks('10 Years')
    years = [d.year for d in breaks(limits)]
    npt.assert_array_equal(years, [2010, 2020, 2030])

    # numpy
    x = [np.datetime64(i*10, 'D') for i in range(1, 10)]
    breaks = date_breaks('10 Years')
    limits = min(x), max(x)
    with pytest.raises(AttributeError):
        breaks(limits)


def test_timedelta_breaks():
    breaks = timedelta_breaks()

    # cpython
    x = [timedelta(days=i*365) for i in range(25)]
    limits = min(x), max(x)
    major = breaks(limits)
    years = [val.total_seconds()/(365*24*60*60)for val in major]
    npt.assert_array_equal(
        years, [0, 5, 10, 15, 20, 25])

    x = [timedelta(microseconds=i) for i in range(25)]
    limits = min(x), max(x)
    major = breaks(limits)
    mseconds = [val.total_seconds()*10**6 for val in major]
    npt.assert_array_equal(
        mseconds, [0, 5, 10, 15, 20, 25])

    # pandas
    x = [pd.Timedelta(seconds=i*60) for i in range(10)]
    limits = min(x), max(x)
    major = breaks(limits)
    minutes = [val.total_seconds()/60 for val in major]
    npt.assert_allclose(
        minutes, [0, 2, 4, 6, 8, 10])

    # numpy
    x = [np.timedelta64(i*10, unit='D') for i in range(1, 10)]
    limits = min(x), max(x)
    with pytest.raises(ValueError):
        breaks(limits)
