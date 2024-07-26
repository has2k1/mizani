"""
Continuous variables have values anywhere in the range minus
infinite to plus infinite. However, when creating a visual
representation of these values what usually matters is the
relative difference between the values. This is where rescaling
comes into play.

The values are mapped onto a range that a scale can deal with. For
graphical representation that range tends to be :math:`[0, 1]` or
:math:`[0, n]`, where :math:`n` is some number that makes the
plotted object overflow the plotting area.

Although a scale may be able handle the :math:`[0, n]` range, it
may be desirable to have a lower bound greater than zero. For
example, if data values get mapped to zero on a scale whose
graphical representation is the size/area/radius/length some data
will be invisible. The solution is to restrict the lower bound
e.g. :math:`[0.1, 1]`. Similarly you can restrict the upper bound
-- using these functions.
"""

from __future__ import annotations

import datetime
import sys
import typing
from copy import copy

import numpy as np
import pandas as pd

from .utils import get_null_value

if typing.TYPE_CHECKING:
    from typing import Any, Optional

    from mizani.typing import (
        FloatArrayLike,
        NDArrayFloat,
        TFloatVector,
        TupleFloat2,
        TupleFloat4,
    )


__all__ = [
    "censor",
    "expand_range",
    "rescale",
    "rescale_max",
    "rescale_mid",
    "squish_infinite",
    "zero_range",
    "expand_range_distinct",
    "squish",
]

EPSILON = sys.float_info.epsilon


def rescale(
    x: FloatArrayLike,
    to: TupleFloat2 = (0, 1),
    _from: Optional[TupleFloat2] = None,
) -> NDArrayFloat:
    """
    Rescale numeric vector to have specified minimum and maximum.

    Parameters
    ----------
    x : array_like | numeric
        1D vector of values to manipulate.
    to : tuple
        output range (numeric vector of length two)
    _from : tuple
        input range (numeric vector of length two).
        If not given, is calculated from the range of x

    Returns
    -------
    out : array_like
        Rescaled values

    Examples
    --------
    >>> x = [0, 2, 4, 6, 8, 10]
    >>> rescale(x)
    array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
    >>> rescale(x, to=(0, 2))
    array([0. , 0.4, 0.8, 1.2, 1.6, 2. ])
    >>> rescale(x, to=(0, 2), _from=(0, 20))
    array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
    """
    __from = (np.min(x), np.max(x)) if _from is None else _from
    return np.interp(x, __from, to)


def rescale_mid(
    x: FloatArrayLike,
    to: TupleFloat2 = (0, 1),
    _from: Optional[TupleFloat2] = None,
    mid: float = 0,
) -> NDArrayFloat:
    """
    Rescale numeric vector to have specified minimum, midpoint,
    and maximum.

    Parameters
    ----------
    x : array_like
        1D vector of values to manipulate.
    to : tuple
        output range (numeric vector of length two)
    _from : tuple
        input range (numeric vector of length two).
        If not given, is calculated from the range of x
    mid	: numeric
        mid-point of input range

    Returns
    -------
    out : array_like
        Rescaled values

    Examples
    --------
    >>> rescale_mid([1, 2, 3], mid=1)
    array([0.5 , 0.75, 1.  ])
    >>> rescale_mid([1, 2, 3], mid=2)
    array([0. , 0.5, 1. ])

    `rescale_mid` does have the same signature as `rescale` and
    `rescale_max`. In cases where we need a compatible function with
    the same signature, we use a closure around the extra `mid` argument.

    >>> def rescale_mid_compat(mid):
    ...     def _rescale(x, to=(0, 1), _from=None):
    ...         return rescale_mid(x, to, _from, mid=mid)
    ...     return _rescale

    >>> rescale_mid2 = rescale_mid_compat(mid=2)
    >>> rescale_mid2([1, 2, 3])
    array([0. , 0.5, 1. ])
    """
    __from: NDArrayFloat = np.array(
        (np.min(x), np.max(x)) if _from is None else _from
    )

    if zero_range(__from) or zero_range(to):  # type: ignore
        out = np.repeat(np.mean(to), len(x))
    else:
        extent = 2 * np.max(np.abs(__from - mid))
        out = (np.asarray(x) - mid) / extent * np.diff(to) + np.mean(to)

    return out


def rescale_max(
    x: FloatArrayLike,
    to: TupleFloat2 = (0, 1),
    _from: Optional[TupleFloat2] = None,
) -> NDArrayFloat:
    """
    Rescale numeric vector to have specified maximum.

    Parameters
    ----------
    x : array_like
        1D vector of values to manipulate.
    to : tuple
        output range (numeric vector of length two)
    _from : tuple
        input range (numeric vector of length two).
        If not given, is calculated from the range of x.
        Only the 2nd (max) element is essential to the
        output.

    Returns
    -------
    out : array_like
        Rescaled values

    Examples
    --------
    >>> x = np.array([0, 2, 4, 6, 8, 10])
    >>> rescale_max(x, (0, 3))
    array([0. , 0.6, 1.2, 1.8, 2.4, 3. ])

    Only the 2nd (max) element of the parameters ``to``
    and ``_from`` are essential to the output.

    >>> rescale_max(x, (1, 3))
    array([0. , 0.6, 1.2, 1.8, 2.4, 3. ])
    >>> rescale_max(x, (0, 20))
    array([ 0.,  4.,  8., 12., 16., 20.])

    If :python:`max(x) < _from[1]` then values will be
    scaled beyond the requested maximum (:python:`to[1]`).

    >>> rescale_max(x, to=(1, 3), _from=(-1, 6))
    array([0., 1., 2., 3., 4., 5.])

    If the values are the same, they taken on the requested maximum.
    This includes an array of all zeros.

    >>> rescale_max(np.array([5, 5, 5]))
    array([1., 1., 1.])
    >>> rescale_max(np.array([0, 0, 0]))
    array([1, 1, 1])
    """
    x = np.asarray(x)
    if _from is None:
        _from = np.min(x), np.max(x)  # type: ignore
        assert _from is not None  # type narrowing

    if np.any(x < 0):
        out = rescale(x, (0, to[1]), _from)
    elif np.all(x == 0) and _from[1] == 0:
        out = np.repeat(to[1], len(x))
    else:
        out = x / _from[1] * to[1]

    return out


def squish_infinite(
    x: FloatArrayLike, range: TupleFloat2 = (0, 1)
) -> NDArrayFloat:
    """
    Truncate infinite values to a range.

    Parameters
    ----------
    x : array_like
        Values that should have infinities squished.
    range : tuple
        The range onto which to squish the infinites.
        Must be of size 2.

    Returns
    -------
    out : array_like
        Values with infinites squished.

    Examples
    --------
    >>> arr1 = np.array([0, .5, .25, np.inf, .44])
    >>> arr2 = np.array([0, -np.inf, .5, .25, np.inf])
    >>> squish_infinite(arr1)
    array([0.  , 0.5 , 0.25, 1.  , 0.44])
    >>> squish_infinite(arr2, (-10, 9))
    array([  0.  , -10.  ,   0.5 ,   0.25,   9.  ])
    """
    _x = np.array(x, copy=True)
    _x[np.isneginf(_x)] = range[0]
    _x[np.isposinf(_x)] = range[1]
    return _x


def squish(
    x: FloatArrayLike, range: TupleFloat2 = (0, 1), only_finite: bool = True
) -> NDArrayFloat:
    """
    Squish values into range.

    Parameters
    ----------
    x : array_like
        Values that should have out of range values squished.
    range : tuple
        The range onto which to squish the values.
    only_finite: boolean
        When true, only squishes finite values.

    Returns
    -------
    out : array_like
        Values with out of range values squished.

    Examples
    --------
    >>> squish([-1.5, 0.2, 0.8, 1.0, 1.2])
    array([0. , 0.2, 0.8, 1. , 1. ])

    >>> squish([-np.inf, -1.5, 0.2, 0.8, 1.0, np.inf], only_finite=False)
    array([0. , 0. , 0.2, 0.8, 1. , 1. ])
    """
    _x = np.array(x, copy=True)
    finite = np.isfinite(_x) if only_finite else True
    _x[np.logical_and(_x < range[0], finite)] = range[0]
    _x[np.logical_and(_x > range[1], finite)] = range[1]
    return _x


def censor(
    x: TFloatVector,
    range: TupleFloat2 = (0, 1),
    only_finite: bool = True,
) -> TFloatVector:
    """
    Convert any values outside of range to a **NULL** type object.

    Parameters
    ----------
    x : array_like
        Values to manipulate
    range : tuple
        (min, max) giving desired output range
    only_finite : bool
        If True (the default), will only modify
        finite values.

    Returns
    -------
    x : array_like
        Censored array

    Examples
    --------
    >>> a = np.array([1, 2, np.inf, 3, 4, -np.inf, 5])
    >>> censor(a, (0, 10))
    array([  1.,   2.,  inf,   3.,   4., -inf,   5.])
    >>> censor(a, (0, 10), False)
    array([ 1.,  2., nan,  3.,  4., nan,  5.])
    >>> censor(a, (2, 4))
    array([ nan,   2.,  inf,   3.,   4., -inf,  nan])

    Notes
    -----
    All values in ``x`` should be of the same type. ``only_finite`` parameter
    is not considered for Datetime and Timedelta types.

    The **NULL** type object depends on the type of values in **x**.

    - :class:`float` - :py:`float('nan')`
    - :class:`int` - :py:`float('nan')`
    - :class:`datetime.datetime` : :py:`np.datetime64(NaT)`
    - :class:`datetime.timedelta` : :py:`np.timedelta64(NaT)`

    """
    res = copy(x)
    if not len(x):
        return res

    null = get_null_value(x)

    if only_finite:
        try:
            finite = np.isfinite(x)
        except TypeError:
            finite = np.repeat(True, len(x))
    else:
        finite = np.repeat(True, len(x))

    # Ignore RuntimeWarning when x contains nans
    with np.errstate(invalid="ignore"):
        outside = (x < range[0]) | (x > range[1])
    bool_idx = finite & outside
    if bool_idx.any():
        if res.dtype == int:
            res = res.astype(float)
        res[bool_idx] = null
    return res


def zero_range(x: tuple[Any, Any], tol: float = EPSILON * 100) -> bool:
    """
    Determine if range of vector is close to zero.

    Parameters
    ----------
    x : array_like
        Value(s) to check. If it is an array_like, it
        should be of length 2.
    tol : float
        Tolerance. Default tolerance is the `machine epsilon`_
        times :math:`10^2`.

    Returns
    -------
    out : bool
        Whether ``x`` has zero range.

    Examples
    --------
    >>> zero_range([1, 1])
    True
    >>> zero_range([1, 2])
    False
    >>> zero_range([1, 2], tol=2)
    True

    .. _machine epsilon: https://en.wikipedia.org/wiki/Machine_epsilon
    """
    if x[0] > x[1]:
        x = x[1], x[0]

    # datetime - pandas, cpython
    if isinstance(x[0], (pd.Timestamp, datetime.datetime)):
        from mizani._core.dates import datetime_to_num

        l, h = datetime_to_num(x)
        return l == h
    # datetime - numpy
    elif isinstance(x[0], np.datetime64):
        return x[0] == x[1]
    # timedelta - pandas, cpython
    elif isinstance(x[0], (pd.Timedelta, datetime.timedelta)):
        return x[0].total_seconds() == x[1].total_seconds()
    # timedelta - numpy
    elif isinstance(x[0], np.timedelta64):
        return x[0] == x[1]
    elif not isinstance(x[0], (float, int, np.number)):
        raise TypeError(
            "zero_range objects cannot work with objects "
            "of type '{}'".format(type(x[0]))
        )
    else:
        low, high = x

    if any(np.isnan((low, high))):
        return True

    if low == high:
        return True

    if any(np.isinf((low, high))):
        return False

    low_abs = np.abs(low)
    if low_abs == 0:
        return False

    return bool(((high - low) / low_abs) < tol)


def expand_range(
    range: TupleFloat2, mul: float = 0, add: float = 0, zero_width: float = 1
) -> TupleFloat2:
    """
    Expand a range with a multiplicative or additive constant

    Parameters
    ----------
    range : tuple
        Range of data. Size 2.
    mul : int | float
        Multiplicative constant
    add : int | float | timedelta
        Additive constant
    zero_width : int | float | timedelta
        Distance to use if range has zero width

    Returns
    -------
    out : tuple
        Expanded range

    Examples
    --------
    >>> expand_range((3, 8))
    (3, 8)
    >>> expand_range((0, 10), mul=0.1)
    (-1.0, 11.0)
    >>> expand_range((0, 10), add=2)
    (-2, 12)
    >>> expand_range((0, 10), mul=.1, add=2)
    (-3.0, 13.0)
    >>> expand_range((0, 1))
    (0, 1)

    When the range has zero width

    >>> expand_range((5, 5))
    (4.5, 5.5)

    Notes
    -----
    If expanding *datetime* or *timedelta* types, **add** and
    **zero_width** must be suitable *timedeltas* i.e. You should
    not mix types between **Numpy**, **Pandas** and the
    :mod:`datetime` module.
    """
    x = range
    low, high = x

    # The expansion cases
    if zero_range(x):
        new = low - zero_width / 2, low + zero_width / 2
    else:
        dx = (high - low) * mul + add
        new = low - dx, high + dx

    return new


def expand_range_distinct(
    range: TupleFloat2,
    expand: TupleFloat2 | TupleFloat4 = (0, 0, 0, 0),
    zero_width: float = 1,
) -> TupleFloat2:
    """
    Expand a range with a multiplicative or additive constants

    Similar to :func:`expand_range` but both sides of the range
    expanded using different constants

    Parameters
    ----------
    range : tuple
        Range of data. Size 2
    expand : tuple
        Length 2 or 4. If length is 2, then the same constants
        are used for both sides. If length is 4 then the first
        two are are the Multiplicative (*mul*) and Additive (*add*)
        constants for the lower limit, and the second two are
        the constants for the upper limit.
    zero_width : int | float | timedelta
        Distance to use if range has zero width

    Returns
    -------
    out : tuple
        Expanded range

    Examples
    --------
    >>> expand_range_distinct((3, 8))
    (3, 8)
    >>> expand_range_distinct((0, 10), (0.1, 0))
    (-1.0, 11.0)
    >>> expand_range_distinct((0, 10), (0.1, 0, 0.1, 0))
    (-1.0, 11.0)
    >>> expand_range_distinct((0, 10), (0.1, 0, 0, 0))
    (-1.0, 10)
    >>> expand_range_distinct((0, 10), (0, 2))
    (-2, 12)
    >>> expand_range_distinct((0, 10), (0, 2, 0, 2))
    (-2, 12)
    >>> expand_range_distinct((0, 10), (0, 0, 0, 2))
    (0, 12)
    >>> expand_range_distinct((0, 10), (.1, 2))
    (-3.0, 13.0)
    >>> expand_range_distinct((0, 10), (.1, 2, .1, 2))
    (-3.0, 13.0)
    >>> expand_range_distinct((0, 10), (0, 0, .1, 2))
    (0, 13.0)
    """

    if len(expand) == 2:
        low_mul = high_mul = expand[0]
        low_add = high_add = expand[1]
    else:
        low_mul, low_add, high_mul, high_add = expand

    lower = expand_range(range, low_mul, low_add, zero_width)[0]
    upper = expand_range(range, high_mul, high_add, zero_width)[1]
    return (lower, upper)
