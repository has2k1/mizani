"""
Limiting data values for a palette
"""

from __future__ import annotations

import datetime
import sys
from copy import copy
from typing import TYPE_CHECKING, overload

import numpy as np

from ._pandas import is_series
from .utils import get_null_value

if TYPE_CHECKING:
    from typing import Any

    import pandas as pd

    from mizani.typing import (
        FloatArrayLike,
        NDArrayFloat,
        TFloatVector,
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
    to: tuple[float, float] = (0, 1),
    _from: tuple[float, float] | None = None,
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
    ```{python}
    from mizani.bounds import rescale

    x = [0, 2, 4, 6, 8, 10]
    rescale(x)
    ```

    ```{python}
    rescale(x, to=(0, 2))
    ```

    ```{python}
    rescale(x, to=(0, 2), _from=(0, 20))
    ```
    """
    __from = (np.nanmin(x), np.nanmax(x)) if _from is None else _from
    return np.interp(x, __from, to)  # pyright: ignore[reportReturnType]


def rescale_mid(
    x: FloatArrayLike,
    to: tuple[float, float] = (0, 1),
    _from: tuple[float, float] | None = None,
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
    ```{python}
    from mizani.bounds import rescale_mid

    rescale_mid([1, 2, 3], mid=1)
    ```

    ```{python}
    rescale_mid([1, 2, 3], mid=2)
    ```

    `rescale_mid` does have the same signature as `rescale` and
    `rescale_max`. In cases where we need a compatible function with
    the same signature, we use a closure around the extra `mid` argument.

    ```{python}
    def rescale_mid_compat(mid):
        def _rescale(x, to=(0, 1), _from=None):
            return rescale_mid(x, to, _from, mid=mid)
        return _rescale
    ```

    ```{python}
    rescale_mid2 = rescale_mid_compat(mid=2)
    rescale_mid2([1, 2, 3])
    ```
    """
    __from: NDArrayFloat = np.array(
        (np.nanmin(x), np.nanmax(x)) if _from is None else _from
    )

    if zero_range(__from) or zero_range(to):  # type: ignore
        out = np.repeat(np.mean(to), len(x))
    else:
        extent = 2 * np.max(np.abs(__from - mid))
        out = (np.asarray(x) - mid) / extent * np.diff(to) + np.mean(to)

    return out


def rescale_max(
    x: FloatArrayLike,
    to: tuple[float, float] = (0, 1),
    _from: tuple[float, float] | None = None,
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
    ```{python}
    import numpy as np

    from mizani.bounds import rescale_max

    x = np.array([0, 2, 4, 6, 8, 10])
    rescale_max(x, (0, 3))
    ```

    Only the 2nd (max) element of the parameters `to`
    and `_from` are essential to the output.

    ```{python}
    rescale_max(x, (1, 3))
    ```

    ```{python}
    rescale_max(x, (0, 20))
    ```

    If `max(x) < _from[1]` then values will be
    scaled beyond the requested maximum (`to[1]`).

    ```{python}
    rescale_max(x, to=(1, 3), _from=(-1, 6))
    ```

    If the values are the same, they taken on the requested maximum.
    This includes an array of all zeros.

    ```{python}
    rescale_max(np.array([5, 5, 5]))
    ```

    ```{python}
    rescale_max(np.array([0, 0, 0]))
    ```
    """
    x = np.asarray(x)
    if _from is None:
        _from = np.nanmin(x), np.nanmax(x)  # type: ignore
        assert _from is not None  # type narrowing

    if np.any(x < 0):
        out = rescale(x, (0, to[1]), _from)
    elif np.all(x == 0) and _from[1] == 0:
        out = np.repeat(to[1], len(x))
    else:
        out = x / _from[1] * to[1]

    return out


def squish_infinite(
    x: FloatArrayLike, range: tuple[float, float] = (0, 1)
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
    ```{python}
    import numpy as np

    from mizani.bounds import squish_infinite

    arr1 = np.array([0, .5, .25, np.inf, .44])
    arr2 = np.array([0, -np.inf, .5, .25, np.inf])
    squish_infinite(arr1)
    ```

    ```{python}
    squish_infinite(arr2, (-10, 9))
    ```
    """
    if is_series(x):
        _x = x.to_numpy().copy().astype(float)
    else:
        _x = np.array(x, copy=True).astype(float)
    _x[np.isneginf(_x)] = range[0]
    _x[np.isposinf(_x)] = range[1]
    return _x


def squish(
    x: FloatArrayLike,
    range: tuple[float, float] = (0, 1),
    only_finite: bool = True,
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
    ```{python}
    import numpy as np

    from mizani.bounds import squish

    squish([-1.5, 0.2, 0.8, 1.0, 1.2])
    ```

    ```{python}
    squish([-np.inf, -1.5, 0.2, 0.8, 1.0, np.inf], only_finite=False)
    ```
    """
    if is_series(x):
        _x = x.to_numpy().copy().astype(float)
    else:
        _x = np.array(x, copy=True).astype(float)
    finite = np.isfinite(_x) if only_finite else True
    _x[np.logical_and(_x < range[0], finite)] = range[0]
    _x[np.logical_and(_x > range[1], finite)] = range[1]
    return _x


def censor(
    x: TFloatVector,
    range: tuple[float, float] = (0, 1),
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

    Notes
    -----
    All values in `x` should be of the same type. `only_finite` parameter
    is not considered for Datetime and Timedelta types.

    The **NULL** type object depends on the type of values in **x**.

    - [](`float`) - `float('nan')`
    - [](`int`) - `float('nan')`
    - [](`datetime.datetime`) : `np.datetime64(NaT)`
    - [](`datetime.timedelta`) : `np.timedelta64(NaT)`

    Examples
    --------
    ```{python}
    import numpy as np

    from mizani.bounds import censor

    a = np.array([1, 2, np.inf, 3, 4, -np.inf, 5])
    censor(a, (0, 10))
    ```

    ```{python}
    censor(a, (0, 10), False)
    ```

    ```{python}
    censor(a, (2, 4))
    ```
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
        res[bool_idx] = null  # pyright: ignore[reportCallIssue,reportArgumentType]
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
        Tolerance. Default tolerance is the
        [machine epsilon](https://en.wikipedia.org/wiki/Machine_epsilon)
        times $10^2$.

    Returns
    -------
    out : bool
        Whether `x` has zero range.

    Examples
    --------
    ```{python}
    from mizani.bounds import zero_range

    zero_range([1, 1])
    ```

    ```{python}
    zero_range([1, 2])
    ```

    ```{python}
    zero_range([1, 2], tol=2)
    ```
    """
    if x[0] > x[1]:
        x = x[1], x[0]

    # datetime - pandas, cpython
    if isinstance(x[0], datetime.datetime):
        from mizani._datetime.utils import datetime_to_num

        l, h = datetime_to_num(x)
        return l == h
    # datetime - numpy
    elif isinstance(x[0], np.datetime64):
        return x[0] == x[1]
    # timedelta - pandas, cpython
    elif isinstance(x[0], datetime.timedelta):
        return x[0].total_seconds() == x[1].total_seconds()
    elif not isinstance(x[0], (float, int, np.number)):
        raise TypeError(
            "zero_range objects cannot work with objects of type '{}'".format(
                type(x[0])
            )
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


@overload
def expand_range(
    range: tuple[float, float],
    mul: float = 0,
    add: float | None = None,
    zero_width: float = 1,
) -> tuple[float, float]: ...


@overload
def expand_range(
    range: tuple[pd.Timestamp, pd.Timestamp],
    mul: float = 0,
    add: pd.Timedelta | None = None,
    zero_width: pd.Timedelta = ...,
) -> tuple[pd.Timestamp, pd.Timestamp]: ...


@overload
def expand_range(
    range: tuple[datetime.datetime, datetime.datetime],
    mul: float = 0,
    add: datetime.timedelta | None = None,
    zero_width: datetime.timedelta = ...,
) -> tuple[datetime.datetime, datetime.datetime]: ...


@overload
def expand_range(
    range: tuple[np.datetime64, np.datetime64],
    mul: float = 0,
    add: np.timedelta64 | None = None,
    zero_width: np.timedelta64 = ...,
) -> tuple[np.datetime64, np.datetime64]: ...


@overload
def expand_range(
    range: tuple[pd.Timedelta, pd.Timedelta],
    mul: float = 0,
    add: pd.Timedelta | None = None,
    zero_width: pd.Timedelta = ...,
) -> tuple[pd.Timedelta, pd.Timedelta]: ...


@overload
def expand_range(
    range: tuple[datetime.timedelta, datetime.timedelta],
    mul: float = 0,
    add: datetime.timedelta | None = None,
    zero_width: datetime.timedelta = ...,
) -> tuple[datetime.timedelta, datetime.timedelta]: ...


@overload
def expand_range(
    range: tuple[np.timedelta64, np.timedelta64],
    mul: float = 0,
    add: np.timedelta64 | None = None,
    zero_width: np.timedelta64 = ...,
) -> tuple[np.timedelta64, np.timedelta64]: ...


def expand_range(
    range: tuple[Any, Any],
    mul: float = 0,
    add: Any | None = None,
    zero_width: Any = 1,
) -> tuple[Any, Any]:
    """
    Expand a range with a multiplicative or additive constant

    Parameters
    ----------
    range : tuple
        Range of data. Size 2.
    mul : int | float
        Multiplicative constant
    add : int | float | timedelta
        Additive constant. If omitted, do not add a constant.
    zero_width : int | float | timedelta
        Distance to use if range has zero width

    Returns
    -------
    out : tuple
        Expanded range

    Notes
    -----
    If expanding *datetime* or *timedelta* types, **add** and
    **zero_width** must be suitable *timedeltas* i.e. You should
    not mix types between **Numpy**, **Pandas** and the
    [](`datetime`) module.

    Examples
    --------
    ```{python}
    from mizani.bounds import expand_range

    expand_range((3, 8))
    ```

    ```{python}
    expand_range((0, 10), mul=0.1)
    ```

    ```{python}
    expand_range((0, 10), add=2)
    ```

    ```{python}
    expand_range((0, 10), mul=.1, add=2)
    ```

    ```{python}
    expand_range((0, 1))
    ```

    When the range has zero width

    ```{python}
    expand_range((5, 5))
    ```
    """
    x = range
    low, high = x

    # The expansion cases
    if zero_range(x):
        new = low - zero_width / 2, low + zero_width / 2
    else:
        dx = (high - low) * mul
        if add is not None:
            dx += add
        new = low - dx, high + dx

    return new


def expand_range_distinct(
    range: tuple[float, float],
    expand: (tuple[float, float] | tuple[float, float, float, float]) = (
        0,
        0,
        0,
        0,
    ),
    zero_width: float = 1,
) -> tuple[float, float]:
    """
    Expand a range with a multiplicative or additive constants

    Similar to [](`~mizani.bounds.expand_range`) but both sides of the range
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
    ```{python}
    from mizani.bounds import expand_range_distinct

    expand_range_distinct((3, 8))
    ```

    ```{python}
    expand_range_distinct((0, 10), (0.1, 0))
    ```

    ```{python}
    expand_range_distinct((0, 10), (0.1, 0, 0.1, 0))
    ```

    ```{python}
    expand_range_distinct((0, 10), (0.1, 0, 0, 0))
    ```

    ```{python}
    expand_range_distinct((0, 10), (0, 2))
    ```

    ```{python}
    expand_range_distinct((0, 10), (0, 2, 0, 2))
    ```

    ```{python}
    expand_range_distinct((0, 10), (0, 0, 0, 2))
    ```

    ```{python}
    expand_range_distinct((0, 10), (.1, 2))
    ```

    ```{python}
    expand_range_distinct((0, 10), (.1, 2, .1, 2))
    ```

    ```{python}
    expand_range_distinct((0, 10), (0, 0, .1, 2))
    ```
    """

    if len(expand) == 2:
        low_mul = high_mul = expand[0]
        low_add = high_add = expand[1]
    else:
        low_mul, low_add, high_mul, high_add = expand

    lower = expand_range(range, low_mul, low_add, zero_width)[0]
    upper = expand_range(range, high_mul, high_add, zero_width)[1]
    return (lower, upper)
