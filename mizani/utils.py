from __future__ import division
from collections import OrderedDict

import numpy as np

from .bounds import zero_range

__all__ = ['seq', 'fullseq', 'round_any', 'min_max', 'match',
           'precision']

DISCRETE_KINDS = 'ObUS'
CONTINUOUS_KINDS = 'ifuc'

SECONDS = OrderedDict([
   ('ns', 1e-9),        # nanosecond
   ('us', 1e-6),        # microsecond
   ('ms', 1e-3),        # millisecond
   ('s', 1),            # second
   ('m', 60),           # month
   ('h', 3600),         # hour
   ('d', 24*3600),      # day
   ('w', 7*24*3600),    # week
   ('M', 31*24*3600),   # month
   ('y', 365*24*3600),  # year
])

NANOSECONDS = OrderedDict([
   ('ns', 1),             # nanosecond
   ('us', 1e3),           # microsecond
   ('ms', 1e6),           # millisecond
   ('s', 1e9),            # second
   ('m', 60e9),           # month
   ('h', 3600e9),         # hour
   ('d', 24*3600e9),      # day
   ('w', 7*24*3600e9),    # week
   ('M', 31*24*3600e9),   # month
   ('y', 365*24*3600e9),  # year
])


def seq(_from=1, to=1, by=1, length_out=None):
    """
    Generate regular sequences

    Parameters
    ----------
    _from : numeric
        start of the sequence.
    to : numeric
        end of the sequence.
    by : numeric
        increment of the sequence.
    length_out : int
        length of the sequence. If a float is supplied, it
        will be rounded up

    Meant to be the same as Rs seq to prevent
    discrepancies at the margins
    """
    if length_out is not None:
        if length_out <= 0:
            raise ValueError(
                "length_out must be greater than zero")
        by = (to - _from)/(np.ceil(length_out)-1)

    x = np.arange(_from, to+by, by)
    if x[-1] > to:
        x = x[:-1]
    return x


def fullseq(range, size, pad=False):
    """
    Generate sequence of fixed size intervals covering range.

    Parameters
    ----------
    range : array_like
        Range of sequence. Must be of length 2
    size : numeric
        interval size
    """
    range = np.asarray(range)
    if zero_range(range):
        return range + size * np.array([-1, 1])/2

    x = seq(
        round_any(range[0], size, np.floor),
        round_any(range[1], size, np.ceil),
        size)

    # Add extra bin on bottom and on top, to guarantee that
    # we cover complete range of data, whether right = True/False
    if pad:
        x = np.hstack([np.min(x) - size, x, np.max(x) + size])
    return x


def round_any(x, accuracy, f=np.round):
    """
    Round to multiple of any number.
    """
    x = np.asarray(x)
    return f(x / accuracy) * accuracy


def min_max(x, nan_rm=False, finite=True):
    """
    Return the minimum and maximum of x

    Parameters
    ----------
    x : array_like
        Sequence
    nan_rm : bool
        Whether to remove ``nan`` values.
    finite : bool
        Whether to consider only finite values.

    Returns
    -------
    out : tuple
        (minimum, maximum) of x
    """
    x = np.asarray(x)
    if nan_rm and finite:
        x = x[np.isfinite(x)]
    elif nan_rm:
        x = x[~np.isnan(x)]
    elif finite:
        x = x[~np.isinf(x)]
    return np.min(x), np.max(x)


def match(v1, v2, nomatch=-1, incomparables=None, start=0):
    """
    Return a vector of the positions of (first)
    matches of its first argument in its second.

    Parameters
    ----------
    v1: array_like
        Values to be matched

    v2: array_like
        Values to be matched against

    nomatch: int
        Value to be returned in the case when
        no match is found.

    incomparables: array_like
        Values that cannot be matched. Any value in ``v1``
        matching a value in this list is assigned the nomatch
        value.
    start: int
        Type of indexing to use. Most likely 0 or 1
    """
    v2_indices = {}
    for i, x in enumerate(v2):
        if x not in v2_indices:
            v2_indices[x] = i

    v1_to_v2_map = [nomatch] * len(v1)
    skip = set(incomparables) if incomparables else set()
    for i, x in enumerate(v1):
        if x in skip:
            continue

        try:
            v1_to_v2_map[i] = v2_indices[x] + start
        except KeyError:
            pass

    return v1_to_v2_map


def precision(x):
    """
    Return the precision of x

    Parameters
    ----------
    x : array_like | numeric
        Value(s) whose for which to compute the precision.

    Returns
    -------
    out : numeric
        The precision of ``x`` or that the values in ``x``.

    Note
    ----
    The precision is computed in base 10.


    >>> precision(0.08)
    0.01
    >>> precision(9)
    1
    >>> precision(16)
    10
    """
    rng = min_max(x, nan_rm=True)
    if zero_range(rng):
        span = np.abs(rng[0])
    else:
        span = np.diff(rng)[0]

    if span == 0:
        return 1
    else:
        return 10 ** int(np.floor(np.log10(span)))
