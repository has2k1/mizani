from __future__ import annotations

import math
import sys
from datetime import datetime, timezone
from typing import TYPE_CHECKING, cast, overload
from warnings import warn

import numpy as np
import pandas as pd
import pandas.api.types as pdtypes

if TYPE_CHECKING:
    from datetime import tzinfo
    from typing import Any, Optional, Sequence, TypeGuard

    from mizani.typing import (
        AnyArrayLike,
        DurationUnit,
        FloatArrayLike,
        FloatSeries,
        NDArrayFloat,
        NullType,
        NumericUFunction,
        SeqDatetime,
        TupleFloat2,
    )


__all__ = [
    "round_any",
    "min_max",
    "match",
    "precision",
    "isclose_abs",
    "same_log10_order_of_magnitude",
    "identity",
    "get_categories",
    "get_timezone",
    "has_dtype",
]

# Use sqrt(epsilon) to correct for loss of precision due floating point
# rounding errors. This value is derived for forward difference numerical
# difference, but for our use cases this choice of value is arbitrary.
EPSILON = sys.float_info.epsilon
ROUNDING_ERROR = np.sqrt(EPSILON)
ABS_TOL = 1e-10  # Absolute Tolerance

DISCRETE_KINDS = "ObUS"
CONTINUOUS_KINDS = "ifuc"

SECONDS: dict[DurationUnit, float] = {
    "ns": 1e-9,  # nanosecond
    "us": 1e-6,  # microsecond
    "ms": 1e-3,  # millisecond
    "s": 1,  # second
    "min": 60,  # minutes
    "h": 3600,  # hour
    "day": 24 * 3600,  # day
    "week": 7 * 24 * 3600,  # week
    "month": 31 * 24 * 3600,  # month
    "year": 365 * 24 * 3600,  # year
}

NANOSECONDS: dict[DurationUnit, float] = {
    "ns": 1,  # nanosecond
    "us": 1e3,  # microsecond
    "ms": 1e6,  # millisecond
    "s": 1e9,  # second
    "min": 60e9,  # minutes
    "h": 3600e9,  # hour
    "day": 24 * 3600e9,  # day
    "week": 7 * 24 * 3600e9,  # week
    "month": 31 * 24 * 3600e9,  # month
    "year": 365 * 24 * 3600e9,  # year
}


@overload
def round_any(
    x: FloatArrayLike, accuracy: float, f: NumericUFunction = np.round
) -> NDArrayFloat: ...


@overload
def round_any(
    x: float, accuracy: float, f: NumericUFunction = np.round
) -> float: ...


def round_any(
    x: FloatArrayLike | float, accuracy: float, f: NumericUFunction = np.round
) -> NDArrayFloat | float:
    """
    Round to multiple of any number.
    """
    if not is_vector(x):
        x = np.asarray(x)
    return f(x / accuracy) * accuracy


def min_max(
    x: FloatArrayLike | float, na_rm: bool = False, finite: bool = True
) -> TupleFloat2:
    """
    Return the minimum and maximum of x

    Parameters
    ----------
    x : array_like
        Sequence
    na_rm : bool
        Whether to remove ``nan`` values.
    finite : bool
        Whether to consider only finite values.

    Returns
    -------
    out : tuple
        (minimum, maximum) of x
    """
    x = np.asarray(x)

    if na_rm and finite:
        x = x[np.isfinite(x)]
    elif not na_rm and np.any(np.isnan(x)):
        return np.nan, np.nan
    elif na_rm:
        x = x[~np.isnan(x)]
    elif finite:
        x = x[~np.isinf(x)]

    if len(x):
        return np.min(x), np.max(x)  # type: ignore
    else:
        return float("-inf"), float("inf")


def match(
    v1: AnyArrayLike,
    v2: AnyArrayLike,
    nomatch: int = -1,
    incomparables: Optional[Any] = None,
    start: int = 0,
) -> list[int]:
    """
    Return a vector of the positions of (first)
    matches of its first argument in its second.

    Parameters
    ----------
    v1: array-like
        The values to be matched

    v2: array-like
        The values to be matched against

    nomatch: int
        The value to be returned in the case when
        no match is found.

    incomparables: array-like
        A list of values that cannot be matched.
        Any value in v1 matching a value in this list
        is assigned the nomatch value.
    start: int
        Type of indexing to use. Most likely 0 or 1
    """
    # NOTE: This function gets called a lot. If it can
    # be optimised, it should.
    lookup: dict[Any, int] = {}
    for i, x in enumerate(v2, start=start):
        if x not in lookup:
            lookup[x] = i

    if incomparables:
        skip = set(incomparables)
        lst = [
            lookup.get(x, nomatch) if x not in skip else nomatch for x in v1
        ]
    else:
        lst = [lookup.get(x, nomatch) for x in v1]
    return lst


def precision(x: FloatArrayLike | float) -> float:
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

    Notes
    -----
    The precision is computed in base 10.

    Examples
    --------
    >>> precision([0.08, 0.09])
    0.01

    Maximum precision is 1

    >>> precision([9, 8])
    1
    >>> precision([16, 78])
    1

    A single values have a precision of 1

    >>> precision(0.08)
    1
    >>> precision([325])
    1
    """

    x = np.asarray(x)
    x = x[~(np.isnan(x) | np.isinf(x))]

    if len(x) <= 1:
        return 1

    smallest_diff = np.diff(np.sort(x))[0]
    if smallest_diff < ROUNDING_ERROR:  # pragma: nocover
        return 1
    else:
        # For some intel processors (Skylake), numpy may be compiled
        # to do a fast but precision losing np.log10 calculation.
        # We add a rounding error incase due to lost precision, a
        # result that should be an integer (or slightly greater than
        # one) will get "floored" to the correct value
        res = 10 ** int(np.floor(np.log10(smallest_diff) + ROUNDING_ERROR) - 1)
        has_extra_zeros = (np.round(x / res) % 10 == 0).all()
        if has_extra_zeros:
            res *= 10

        # 1 comes first so that min(1, 1.0) returns an integer
        return min(1, res)


def same_log10_order_of_magnitude(x, delta=0.1):
    """
    Return true if range is approximately in same order of magnitude

    For example these sequences are in the same order of magnitude:

        - [1, 8, 5]     # [1, 10)
        - [35, 20, 80]  # [10 100)
        - [232, 730]    # [100, 1000)

    Parameters
    ----------
    x : array-like
         Values in base 10. Must be size 2 and
        ``rng[0] <= rng[1]``.
    delta : float
        Fuzz factor for approximation. It is multiplicative.
    """
    dmin = np.log10(np.min(x) * (1 - delta))
    dmax = np.log10(np.max(x) * (1 + delta))
    return np.floor(dmin) == np.floor(dmax)


def identity(*args):
    """
    Return whatever is passed in
    """
    return args if len(args) > 1 else args[0]


def get_categories(x):
    """
    Return the categories of x

    Parameters
    ----------
    x : category_like
        Input Values

    Returns
    -------
    out : Index
        Categories of x
    """
    try:
        return x.cat.categories  # series
    except AttributeError:
        try:
            return x.categories  # plain categorical
        except AttributeError as err:
            raise TypeError("x is not a categorical") from err


def log(x, base):
    """
    Calculate the log of x

    Parameters
    ----------
    x : category_like
        Input Values

    base : float
        Base of logarithm

    Returns
    -------
    out : float
        Log of x
    """
    if base == 10:
        res = np.log10(x)
    elif base == 2:
        res = np.log2(x)
    elif base == np.e:
        res = np.log(x)
    else:
        res = np.log(x) / np.log(base)
    return res


def get_timezone(x: SeqDatetime) -> tzinfo | None:
    """
    Return a single timezone for the sequence of datetimes

    Returns the timezone of first item and warns if any other items
    have a different timezone
    """

    # Ref: https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
    x0 = next(iter(x))
    if not isinstance(x0, datetime):
        return None

    x = cast(list[datetime], x)
    info = x0.tzinfo
    if info is None:
        return timezone.utc

    # Consistency check
    tzname0 = info.tzname(x0)
    tznames = (dt.tzinfo.tzname(dt) if dt.tzinfo else None for dt in x)

    if any(tzname0 != name for name in tznames):
        msg = (
            "Dates in column have different time zones. "
            f"Choosen `{tzname0}` the time zone of the first date. "
            "To use a different time zone, create a "
            "labeller and pass the time zone."
        )
        warn(msg)
    return info


def get_null_value(x: Any) -> NullType:
    """
    Return a Null value for the type of values
    """
    from datetime import datetime, timedelta

    import pandas as pd

    x0 = next(iter(x))
    numeric_types: Sequence[type] = (np.int64, np.float64, int, float, bool)

    if pdtypes.is_object_dtype(x):
        return None
    elif isinstance(x0, numeric_types):  # type: ignore
        return float("nan")
    # pandas types subclass cypthon types, so check
    # for them first
    elif isinstance(x0, (pd.Timestamp, pd.Timedelta)):
        return type(x0)("NaT")
    elif isinstance(x0, (datetime, timedelta)):
        return None
    elif isinstance(x0, (np.datetime64, np.timedelta64)):
        return type(x0)("NaT")
    else:
        raise ValueError(
            "Cannot get a null value for type: {}".format(type(x[0]))
        )


def isclose_abs(a: float, b: float, tol: float = ABS_TOL) -> bool:
    """
    Return True if a and b are close given the absolute tolerance
    """
    return math.isclose(a, b, rel_tol=0, abs_tol=ABS_TOL)


def is_vector(x: Any) -> TypeGuard[NDArrayFloat | FloatSeries]:
    """
    Return True if x is a numpy array or a pandas series
    """
    return isinstance(x, (np.ndarray, pd.Series))


def has_dtype(
    x: Any,
) -> TypeGuard[np.ndarray | pd.Series | pd.Categorical | pd.Index]:
    """
    Return True if x has the dtype property
    """
    return hasattr(x, "dtype")
