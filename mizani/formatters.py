"""
Scales have guides and these are what help users make sense of
the data mapped onto the scale. Common examples of guides include
the x-axis, the y-axis, the keyed legend and a colorbar legend.
The guides have demarcations(breaks), some of which must be labelled.

The `*_format` functions below create functions that convert data
values as understood by a specific scale and return string
representations of those values. Manipulating the string
representation of a value helps improve readability of the guide.
"""
from __future__ import division
import re

import numpy as np
from matplotlib.dates import DateFormatter
from matplotlib.ticker import ScalarFormatter

from .breaks import timedelta_helper
from .utils import round_any, precision


__all__ = ['custom_format', 'currency_format', 'dollar_format',
           'percent_format', 'scientific_format', 'date_format',
           'mpl_format', 'timedelta_format']


def custom_format(fmt, style='new'):
    """
    Return a function that formats a sequence of inputs

    Parameters
    ----------
    fmt : str
        Format string
    style : 'new' | 'old'
        Whether to use new style or old style formatting.
        New style uses the :meth:`str.format` while old
        style uses ``%``. The format string must be written
        accordingly.

    Returns
    -------
    out : function
        Formatting function. It takes a sequence of
        values and returns a sequence of strings.


    >>> formatter = custom_format('{:.2f} USD')
    >>> formatter([3.987, 2, 42.42])
    ['3.99 USD', '2.00 USD', '42.42 USD']
    """
    def _custom_format(x):
        if style == 'new':
            return [fmt.format(val) for val in x]
        elif style == 'old':
            return [fmt % val for val in x]
        else:
            raise ValueError(
                "style should be either 'new' or 'old'")

    return _custom_format


# formatting functions
def currency_format(prefix='$', suffix='',
                    digits=2, big_mark=''):
    """
    Currency formatter

    Parameters
    ----------
    prefix : str
        What to put before the value.
    suffix : str
        What to put after the value.
    digits : int
        Number of significant digits
    big_mark : str
        The thousands separator. This is usually
        a comma or a dot.

    Returns
    -------
    out : function
        Formatting function. It takes a sequence of
        numerical values and and returns a sequence
        of strings.


    >>> x = [1.232, 99.2334, 4.6, 9, 4500]
    >>> currency_format()(x)
    ['$1.23', '$99.23', '$4.60', '$9.00', '$4500.00']
    >>> currency_format('C$', digits=0, big_mark=',')(x)
    ['C$1', 'C$99', 'C$5', 'C$9', 'C$4,500']
    """
    # create {:.2f} or {:,.2f}
    bm = ',' if big_mark else ''
    tpl = ''.join((prefix, '{:', bm, '.',
                   str(digits), 'f}', suffix))

    def _currency_format(x):
        labels = [tpl.format(val) for val in x]
        if big_mark and big_mark != ',':
            labels = [val.replace(',', big_mark) for val in labels]
        return labels

    return _currency_format


dollar_format = currency_format
dollar = dollar_format()


def comma_format(digits=0):
    """
    Format number with commas separating thousands

    Parameters
    ----------
    digits : int
        Number of digits after the decimal point.

    Returns
    -------
    out : function
        Formatting function. It takes a sequence of
        numerical values and and returns a sequence
        of strings.


    >>> comma_format()([1000, 2, 33000, 400])
    ['1,000', '2', '33,000', '400']
    """
    formatter = currency_format(prefix='',
                                digits=digits,
                                big_mark=',')

    def _comma_format(x):
        return formatter(x)

    return _comma_format


def percent_format(use_comma=False):
    """
    Percent formatter

    Multiply by one hundred and display percent sign

    Returns
    -------
    out : function
        Formatting function. It takes a sequence of
        numerical values and and returns a sequence
        of strings.


    >>> formatter = percent_format()
    >>> formatter([.45, 9.515, .01])
    ['45%', '952%', '1%']
    >>> formatter([.654, .8963, .1])
    ['65.4%', '89.6%', '10.0%']
    """
    # unnecessary zeros
    zeros_re = re.compile('\.0+%$')
    big_mark = ',' if use_comma else ''

    def _percent_format(x):
        _precision = precision(x)
        x = round_any(x, _precision / 100) * 100

        # When the precision is less than 1, we show
        if _precision > 1:
            digits = 0
        else:
            digits = abs(int(np.log10(_precision)))

        formatter = currency_format(prefix='',
                                    suffix='%',
                                    digits=digits,
                                    big_mark=big_mark)
        labels = formatter(x)
        # Remove unnecessary zeros after the decimal
        if all(zeros_re.search(val) for val in labels):
            labels = [zeros_re.sub('%', val) for val in labels]
        return labels

    return _percent_format


percent = percent_format()


def scientific_format(digits=3):
    """
    Scientific formatter

    Parameters
    ----------
    digits : int
        Significant digits.

    Returns
    -------
    out : function
        Formatting function. It takes a sequence of
        values and returns a sequence of strings.


    >>> x = [.12, .23, .34, 45]
    >>> scientific_format()(x)
    ['1.2e-01', '2.3e-01', '3.4e-01', '4.5e+01']

    Note
    ----
    Be careful when using many digits (15+ on a 64
    bit computer). Consider of the `machine epsilon`_.

    .. _machine epsilon: https://en.wikipedia.org/wiki/Machine_epsilon
    """
    zeros_re = re.compile(r'(0+)e')
    tpl = ''.join(['{:.', str(digits), 'e}'])
    formatter = custom_format(tpl)

    def count_zeros(s):
        match = zeros_re.search(s)
        if match:
            return len(match.group(1))
        else:
            return 0

    def _scientific_format(x):
        # format and then remove superfluous zeros
        labels = formatter(x)
        n = min([count_zeros(val) for val in labels])
        if n:
            labels = [val.replace('0'*n+'e', 'e') for val in labels]
        return labels

    return _scientific_format


scientific = scientific_format()


def _format(formatter, x):
    """
    Helper to format and tidy up
    """
    # For MPL to play nice
    formatter.create_dummy_axis()
    # For sensible decimal places
    formatter.set_locs([val for val in x if ~np.isnan(val)])
    try:
        oom = int(formatter.orderOfMagnitude)
    except AttributeError:
        oom = 0
    labels = [formatter(tick) for tick in x]

    # Remove unnecessary decimals
    pattern = re.compile(r'\.0+$')
    for i, label in enumerate(labels):
        match = re.search(pattern, label)
        if match:
            labels[i] = re.sub(pattern, '', label)

    # MPL does not add the exponential component
    if oom:
        labels = ['{}e{}'.format(s, oom) if s != '0' else s
                  for s in labels]
    return labels


def mpl_format():
    """
    Format using MPL formatter for scalars

    Returns
    -------
    out : function
        Formatting function. It takes a sequence of
        values and returns a sequence of strings.


    >>> mpl_format()([.654, .8963, .1])
    ['0.6540', '0.8963', '0.1000']
    """
    formatter = ScalarFormatter(useOffset=False)

    def _mpl_format(x):
        return _format(formatter, x)

    return _mpl_format


def date_format(fmt='%Y-%m-%d'):
    """
    Datetime formatter

    Parameters
    ----------
    fmt : str
        Format string. See
        :ref:`strftime <strftime-strptime-behavior>`.

    Returns
    -------
    out : function
        Formatting function. It takes a sequence of
        values and returns a sequence of strings.


    >>> from datetime import datetime
    >>> x = [datetime(x, 1, 1) for x in [2010, 2014, 2018, 2022]]
    >>> date_format()(x)
    ['2010-01-01', '2014-01-01', '2018-01-01', '2022-01-01']
    >>> date_format('%Y')(x)
    ['2010', '2014', '2018', '2022']
    """
    formatter = DateFormatter(fmt)

    def _date_format(x):
        # The formatter is tied to axes and takes
        # breaks in ordinal format.
        x = [val.toordinal() for val in x]
        return _format(formatter, x)

    return _date_format


def timedelta_format(units=None, add_units=True, usetex=False):
    """
    Timedelta formatter

    Returns
    -------
    out : function
        Formatting function. It takes a sequence of
        timedelta values and returns a sequence of
        strings.

    >>> from datetime import timedelta
    >>> x = [timedelta(days=31*i) for i in range(5)]
    >>> timedelta_format()(x)
    ['0', '1 month', '2 months', '3 months', '4 months']
    >>> timedelta_format(units='d')(x)
    ['0', '31 days', '62 days', '93 days', '124 days']
    """
    abbreviations = {
        'ns': 'ns',
        'us': '$\mu s$' if usetex else 'us',
        'ms': 'ms',
        's': 's',
        'm': ' minute',
        'h': ' hour',
        'd': ' day',
        'w': ' week',
        'M': ' month',
        'y': ' year'}
    _mpl_format = mpl_format()

    def _timedelta_format(x):
        labels = []
        values, _units = timedelta_helper.format_info(x, units)
        plural = '' if _units.endswith('s') else 's'
        ulabel = abbreviations[_units]
        _labels = _mpl_format(values)

        for num, num_label in zip(values, _labels):
            s = '' if num == 1 else plural
            # 0 has no units
            _ulabel = '' if num == 0 else ulabel+s
            labels.append(''.join([num_label, _ulabel]))

        return labels

    return _timedelta_format
