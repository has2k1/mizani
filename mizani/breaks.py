"""
All scales have a means by which the values that are mapped
onto the scale are interpreted. Numeric digital scales put
out numbers for direct interpretation, but most scales
cannot do this. What they offer is named markers/ticks that
aid in assessing the values e.g. the common odometer will
have ticks and values to help gauge the speed of the vehicle.

The named markers are what we call breaks. Properly calculated
breaks make interpretation straight forward. These functions
provide ways to calculate good(hopefully) breaks.
"""
from __future__ import division

import numpy as np
from matplotlib.dates import MinuteLocator, HourLocator, DayLocator
from matplotlib.dates import WeekdayLocator, MonthLocator, YearLocator
from matplotlib.dates import AutoDateLocator
from matplotlib.dates import num2date, YEARLY
from matplotlib.ticker import MaxNLocator

from .utils import min_max, SECONDS, NANOSECONDS


__all__ = ['mpl_breaks', 'log_breaks', 'minor_breaks',
           'trans_minor_breaks', 'date_breaks',
           'timedelta_breaks']


# The break calculations rely on MPL locators to do
# the heavylifting. It may be more convinient to lift
# the calculations out of MPL.

class DateLocator(AutoDateLocator):

    def __init__(self):
        AutoDateLocator.__init__(self, minticks=5,
                                 interval_multiples=True)
        self.intervald[YEARLY] = [5, 50]
        self.create_dummy_axis()

    def tick_values(self, vmin, vmax):
        ticks = AutoDateLocator.tick_values(self, vmin, vmax)
        return ticks


def mpl_breaks(*args, **kwargs):
    """
    Compute breaks using MPL's default locator

    See :class:`~matplotlib.ticker.MaxNLocator` for the
    parameter descriptions

    Returns
    -------
    out : function
        A function that takes a sequence of values and
        returns a sequence of break points.


    >>> x = range(10)
    >>> limits = (0, 9)
    >>> mpl_breaks()(limits)
    array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.])
    >>> mpl_breaks(nbins=2)(limits)
    array([  0.,   5.,  10.])
    """
    locator = MaxNLocator(*args, **kwargs)

    def _mpl_breaks(x):
        return locator.tick_values(np.min(x), np.max(x))

    return _mpl_breaks


def log_breaks(n=5, base=10):
    """
    Integer breaks on log transformed scales

    Parameters
    ----------
    n : int
        Desired number of breaks
    base : int
        Base of logarithm

    Returns
    -------
    out : function
        A function that takes a tuple with 2 limit
        values and returns a sequence of break points.


    >>> x = np.logspace(3, 7)
    >>> limits = min(x), max(x)
    >>> log_breaks()(limits)
    array([    100,   10000, 1000000])
    >>> log_breaks(2)(limits)
    array([   100, 100000])
    """
    def _log_breaks(limits):
        rng = np.log(limits)/np.log(base)
        _min = int(np.floor(rng[0]))
        _max = int(np.ceil(rng[1]))

        if _max == _min:
            return base ** _min

        step = (_max-_min)//n + 1
        return base ** np.arange(_min, _max+1, step)

    return _log_breaks


def minor_breaks(n=1):
    """
    Compute minor breaks

    Parameters
    ----------
    n : int
        Number of minor breaks between the major
        breaks.

    Returns
    -------
    out : function
        Function that computes minor breaks.


    >>> major = [1, 2, 3, 4]
    >>> limits = [0, 5]
    >>> minor_breaks()(major, limits)
    array([ 0.5,  1.5,  2.5,  3.5,  4.5])
    """
    def _minor_breaks(major, limits=None):
        """
        Minor breaks

        Parameters
        ----------
        major : array_like
            Major breaks
        limits : array_like | None
            Limits of the scale. If *array_like*, must be
            of size 2. If **None**, then the minimum and
            maximum of the major breaks are used.
        """
        if len(major) < 2:
            return np.array([])

        if limits is None:
            limits = min_max(major)

        # Try to infer additional major breaks so that
        # minor breaks can be generated beyond the first
        # and last major breaks
        diff = np.diff(major)
        step = diff[0]
        if len(diff) > 1 and all(diff == step):
            major = np.hstack([major[0]-step,
                               major,
                               major[-1]+step])

        mbreaks = []
        factors = np.arange(1, n+1)
        for lhs, rhs in zip(major[:-1], major[1:]):
            sep = (rhs - lhs)/(n+1)
            mbreaks.append(lhs + factors * sep)

        minor = np.hstack(mbreaks)
        minor = minor.compress((limits[0] <= minor) &
                               (minor <= limits[1]))
        return minor

    return _minor_breaks


def trans_minor_breaks(n=1):
    """
    Compute minor breaks for transformed scales

    The minor breaks are computed in data space.
    This together with major breaks computed in
    transform space reveals the non linearity of
    of a scale. See the log transforms created
    with :func:`log_trans` like :class:`log10_trans`.

    Parameters
    ----------
    n : int
        Number of minor breaks between the major
        breaks.

    Returns
    -------
    out : classmethod
        Method for the transform class that
        computes minor breaks.


    >>> from mizani.transforms import sqrt_trans
    >>> major = [1, 2, 3, 4]
    >>> limits = [0, 5]
    >>> sqrt_trans.minor_breaks(major, limits)
    array([ 0.5,  1.5,  2.5,  3.5,  4.5])
    >>> class sqrt_trans2(sqrt_trans):
    ...     minor_breaks = trans_minor_breaks()
    >>> sqrt_trans2.minor_breaks(major, limits)
    array([ 1.58113883,  2.54950976,  3.53553391])
    """
    _minor_breaks = minor_breaks(n)

    @classmethod
    def _trans_minor_breaks(trans, major, limits=None):
        """
        Minor breaks for transformed scales

        Parameters
        ----------
        major : array_like
            Major breaks
        limits : array_like | None
            Limits of the scale. If *array_like*, must be
            of size 2. If **None**, then the minimum and
            maximum of the major breaks are used.
        """
        if not trans.dataspace_is_numerical:
            raise TypeError(
                "trans_minor_breaks can only be used for data"
                "whose format is numerical.")

        if limits is None:
            limits = min_max(major)

        major = _extend_trans_breaks(trans, major)
        major = trans.inverse(major)
        limits = trans.inverse(limits)
        minor = _minor_breaks(major, limits)
        return trans.transform(minor)

    return _trans_minor_breaks


def _extend_trans_breaks(trans, major):
    """
    Append 2 extra breaks at either end of major

    If breaks of transform space are non-equidistant,
    :func:`minor_breaks` add minor breaks beyond the first
    and last major breaks. The solutions is to extend those
    breaks (in transformed space) before the minor break call
    is made. How the breaks depends on the type of transform.
    """
    trans = trans if isinstance(trans, type) else trans.__class__
    # so far we are only certain about this extending stuff
    # making sense for log transform
    is_log = trans.__name__.startswith('log')
    diff = np.diff(major)
    step = diff[0]
    if is_log and all(diff == step):
        major = np.hstack([major[0]-step, major, major[-1]+step])
    return major


# Matplotlib's YearLocator uses different named
# arguments than the others
LOCATORS = {
    'minute': MinuteLocator,
    'hour': HourLocator,
    'day': DayLocator,
    'week': WeekdayLocator,
    'month': MonthLocator,
    'year': lambda interval: YearLocator(base=interval)
}


def date_breaks(width=None):
    """
    Regularly spaced dates

    Parameters
    ----------
    width: str | None
        An interval specification. Must be one of
        [minute, hour, day, week, month, year]
        If ``None``, the interval automatic.

    Returns
    -------
    out : function
        A function that takes a tuple of two
        :class:`datetime.datetime` values and returns
        a sequence of break points.


    >>> from datetime import datetime
    >>> x = [datetime(year, 1, 1) for year in [2010, 2026, 2015]]

    Default breaks will be regularly spaced but the spacing
    is automatically determined

    >>> limits = min(x), max(x)
    >>> breaks = date_breaks()
    >>> [d.year for d in breaks(limits)]
    [2014, 2019, 2024]

    Breaks at 4 year intervals

    >>> breaks = date_breaks('4 year')
    >>> [d.year for d in breaks(limits)]
    [2008, 2012, 2016, 2020, 2024, 2028]
    """
    if not width:
        locator = DateLocator()
    else:
        # Parse the width specification
        # e.g. '10 weeks' => (10, week)
        _n, units = width.strip().lower().split()
        interval, units = int(_n), units.rstrip('s')
        locator = LOCATORS[units](interval=interval)

    def _date_breaks(limits):
        ret = locator.tick_values(*limits)
        # MPL returns the tick_values in ordinal format,
        # but we return them in the same space as the
        # inputs.
        return [num2date(val) for val in ret]

    return _date_breaks


def timedelta_breaks():
    """
    Timedelta breaks

    Returns
    -------
    out : function
        A function that takes a sequence of two
        :class:`datetime.timedelta` values and returns
        a sequence of break points.


    >>> from datetime import timedelta
    >>> breaks = timedelta_breaks()
    >>> x = [timedelta(days=i*365) for i in range(25)]
    >>> limits = min(x), max(x)
    >>> major = breaks(limits)
    >>> [val.total_seconds()/(365*24*60*60)for val in major]
    [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
    """
    _mpl_breaks = mpl_breaks(nbins=7, steps=(1, 2, 5, 10))

    def _timedelta_breaks(limits):
        helper = timedelta_helper(limits)
        scaled_limits = helper.scaled_limits()
        scaled_breaks = _mpl_breaks(scaled_limits)
        breaks = helper.numeric_to_timedelta(scaled_breaks)
        return breaks

    return _timedelta_breaks


# This could be cleaned up, state overload?
class timedelta_helper(object):
    """
    Helper for computing timedelta breaks
    and labels.

    How to use - breaks?

    1. Initialise with a timedelta sequence/limits.
    2. Get the scaled limits and use those to calculate
       breaks using a general purpose breaks calculating
       routine. The scaled limits are in numerical format.
    3. Convert the computed breaks from numeric into timedelta.

    See, :func:`timedelta_breaks`

    How to use - formating?

    1. Call :meth:`format_info` with the timedelta values to be
       formatted and get back a tuple of numeric values and
       the units for those values.
    2. Format the values with a general purpose formatting
       routing.

    See, :func:`timedelta_format`
    """
    def __init__(self, x, units=None):
        self.x = x
        self.type = type(x[0])
        self.package = self.determine_package(x[0])
        _limits = min(x), max(x)
        self.limits = self.value(_limits[0]), self.value(_limits[1])
        self.units = units or self.best_units(_limits)
        self.factor = self.get_scaling_factor(self.units)

    @classmethod
    def determine_package(cls, td):
        if hasattr(td, 'components'):
            package = 'pandas'
        elif hasattr(td, 'total_seconds'):
            package = 'cpython'
        else:
            msg = '{} format not yet supported.'
            raise ValueError(msg.format(td.__class__))
        return package

    @classmethod
    def format_info(cls, x, units=None):
        helper = cls(x, units)
        return helper.timedelta_to_numeric(x), helper.units

    def best_units(self, sequence):
        """
        Determine good units for representing a sequence of timedeltas
        """
        # Read
        #   [(0.9, 's'),
        #    (9, 'm)]
        # as, break ranges between 0.9 seconds (inclusive)
        # and 9 minutes are represented in seconds. And so on.
        ts_range = self.value(max(sequence)) - self.value(min(sequence))
        package = self.determine_package(sequence[0])
        if package == 'pandas':
            cuts = [
                (0.9, 'us'),
                (0.9, 'ms'),
                (0.9, 's'),
                (9, 'm'),
                (6, 'h'),
                (4, 'd'),
                (4, 'w'),
                (4, 'M'),
                (3, 'y')]
            denomination = NANOSECONDS
            base_units = 'ns'
        else:
            cuts = [
                (0.9, 's'),
                (9, 'm'),
                (6, 'h'),
                (4, 'd'),
                (4, 'w'),
                (4, 'M'),
                (3, 'y')]
            denomination = SECONDS
            base_units = 'ms'

        for size, units in reversed(cuts):
            if ts_range >= size*denomination[units]:
                return units

        return base_units

    def value(self, td):
        """
        Return the numeric value representation on a timedelta
        """
        if self.package == 'pandas':
            return td.value
        else:
            return td.total_seconds()

    def scaled_limits(self):
        """
        Minimum and Maximum to use for computing breaks
        """
        _min = self.limits[0]/self.factor
        _max = self.limits[1]/self.factor
        return _min, _max

    def timedelta_to_numeric(self, timedeltas):
        """
        Convert sequence of timedelta to numerics
        """
        return [self.to_numeric(td) for td in timedeltas]

    def numeric_to_timedelta(self, numerics):
        """
        Convert sequence of numerics to timedelta
        """
        if self.package == 'pandas':
            return [self.type(int(x*self.factor), units='ns')
                    for x in numerics]
        else:
            return [self.type(seconds=x*self.factor)
                    for x in numerics]

    def get_scaling_factor(self, units):
        if self.package == 'pandas':
            return NANOSECONDS[units]
        else:
            return SECONDS[units]

    def to_numeric(self, td):
        """
        Convert timedelta to a number corresponding to the
        appropriate units. The appropriate units are those
        determined with the object is initialised.
        """
        if self.package == 'pandas':
            return td.value/NANOSECONDS[self.units]
        else:
            return td.total_seconds()/SECONDS[self.units]
