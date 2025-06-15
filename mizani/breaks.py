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

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date, datetime
from itertools import product
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .utils import (
    log,
    min_max,
    round_any,
)

if TYPE_CHECKING:
    from typing import Sequence

    from mizani.typing import (
        DatetimeOffset,
        FloatArrayLike,
        NDArrayFloat,
        Timedelta,
        TimedeltaArrayLike,
        TimedeltaOffset,
        Trans,
    )


__all__ = [
    "breaks_log",
    "breaks_symlog",
    "minor_breaks",
    "minor_breaks_trans",
    "breaks_date",
    "breaks_date_width",
    "breaks_width",
    "breaks_timedelta",
    "breaks_timedelta_width",
    "breaks_extended",
]


@dataclass
class breaks_log:
    """
    Integer breaks on log transformed scales

    Parameters
    ----------
    n : int
        Desired number of breaks
    base : int
        Base of logarithm

    Examples
    --------
    >>> x = np.logspace(3, 6)
    >>> limits = min(x), max(x)
    >>> breaks_log()(limits)
    array([     1000,    10000,   100000,  1000000])
    >>> breaks_log(2)(limits)
    array([  1000, 100000])
    >>> breaks_log()([0.1, 1])
    array([0.1, 0.3, 1. , 3. ])
    """

    n: int = 5
    base: float = 10

    def __call__(self, limits: tuple[float, float]) -> NDArrayFloat:
        """
        Compute breaks

        Parameters
        ----------
        limits : tuple
            Minimum and maximum values

        Returns
        -------
        out : array_like
            Sequence of breaks points
        """
        if any(np.isinf(limits)):
            return np.array([])

        n = self.n
        base = self.base
        rng = log(limits, base)
        _min = int(np.floor(rng[0]))
        _max = int(np.ceil(rng[1]))

        # Prevent overflow
        if float(base) ** _max > sys.maxsize:
            base = float(base)

        if _max == _min:
            return np.array([base**_min])

        # Try getting breaks at the integer powers of the base
        # e.g [1, 100, 10000, 1000000]
        # If there are too few breaks, try other points using the
        # _log_sub_breaks
        by = int(np.floor((_max - _min) / n)) + 1
        for step in range(by, 0, -1):
            breaks = np.array([base**i for i in range(_min, _max + 1, step)])
            relevant_breaks = (limits[0] <= breaks) & (breaks <= limits[1])
            if np.sum(relevant_breaks) >= n - 2:
                return breaks

        return _breaks_log_sub(n=n, base=base)(limits)


@dataclass
class _breaks_log_sub:
    """
    Breaks for log transformed scales

    Calculate breaks that do not fall on integer powers of
    the base.

    Parameters
    ----------
    n : int
        Desired number of breaks
    base : int | float
        Base of logarithm

    Notes
    -----
    Credit: Thierry Onkelinx (thierry.onkelinx@inbo.be) for the original
    algorithm in the r-scales package.
    """

    n: int = 5
    base: float = 10

    def __call__(self, limits: tuple[float, float]) -> NDArrayFloat:
        base = self.base
        n = self.n
        rng = log(limits, base)
        _min = int(np.floor(rng[0]))
        _max = int(np.ceil(rng[1]))
        steps = [1]

        # Prevent overflow
        if float(base) ** _max > sys.maxsize:
            base = float(base)

        def delta(x):
            """
            Calculates the smallest distance in the log scale between the
            currently selected breaks and a new candidate 'x'
            """
            arr = np.sort(np.hstack([x, steps, base]))
            log_arr = log(arr, base)
            return np.min(np.diff(log_arr))

        if self.base == 2:
            return np.array([base**i for i in range(_min, _max + 1)])

        candidate = np.arange(base + 1)
        candidate = np.compress(
            (candidate > 1) & (candidate < base), candidate
        )

        while len(candidate):
            best = np.argmax([delta(x) for x in candidate])
            steps.append(candidate[best])
            candidate = np.delete(candidate, best)
            _factors = [base**i for i in range(_min, _max + 1)]
            breaks = np.array([f * s for f, s in product(_factors, steps)])
            relevant_breaks = (limits[0] <= breaks) & (breaks <= limits[1])

            if np.sum(relevant_breaks) >= n - 2:
                breaks = np.sort(breaks)
                lower_end = np.max(
                    [
                        np.min(np.where(limits[0] <= breaks)) - 1,
                        0,  # type: ignore
                    ]
                )
                upper_end = np.min(
                    [
                        np.max(np.where(breaks <= limits[1])) + 1,
                        len(breaks),  # type: ignore
                    ]
                )
                return breaks[lower_end : upper_end + 1]
        else:
            return breaks_extended(n=n)(limits)


@dataclass
class minor_breaks:
    """
    Compute minor breaks

    This is the naive method. It does not take into account
    the transformation.

    Parameters
    ----------
    n : int
        Number of minor breaks between the major
        breaks.

    Examples
    --------
    >>> major = [1, 2, 3, 4]
    >>> limits = [0, 5]
    >>> minor_breaks()(major, limits)
    array([0.5, 1.5, 2.5, 3.5, 4.5])
    >>> minor_breaks()([1, 2], (1, 2))
    array([1.5])

    More than 1 minor break.

    >>> minor_breaks(3)([1, 2], (1, 2))
    array([1.25, 1.5 , 1.75])
    >>> minor_breaks()([1, 2], (1, 2), 3)
    array([1.25, 1.5 , 1.75])
    """

    n: int = 1

    def __call__(
        self,
        major: FloatArrayLike,
        limits: tuple[float, float] | None = None,
        n: int | None = None,
    ) -> NDArrayFloat:
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
        n : int
            Number of minor breaks between the major
            breaks. If **None**, then *self.n* is used.

        Returns
        -------
        out : array_like
            Minor beraks
        """
        if len(major) < 2:
            return np.array([])

        if limits is None:
            low, high = min_max(major)
        else:
            low, high = min_max(limits)

        if n is None:
            n = self.n

        # Try to infer additional major breaks so that
        # minor breaks can be generated beyond the first
        # and last major breaks
        diff = np.diff(major)
        step = diff[0]
        if len(diff) > 1 and all(diff == step):
            major = np.hstack([major[0] - step, major, major[-1] + step])

        mbreaks = []
        factors = np.arange(1, n + 1)
        for lhs, rhs in zip(major[:-1], major[1:]):
            sep = (rhs - lhs) / (n + 1)
            mbreaks.append(lhs + factors * sep)

        minor = np.hstack(mbreaks)
        minor = minor.compress((low <= minor) & (minor <= high))
        return minor


@dataclass
class minor_breaks_trans:
    """
    Compute minor breaks for transformed scales

    The minor breaks are computed in data space.
    This together with major breaks computed in
    transform space reveals the non linearity of
    of a scale. See the log transforms created
    with :func:`log_trans` like :class:`log10_trans`.

    Parameters
    ----------
    trans : trans or type
        Trans object or trans class.
    n : int
        Number of minor breaks between the major
        breaks.

    Examples
    --------
    >>> from mizani.transforms import sqrt_trans
    >>> major = [1, 2, 3, 4]
    >>> limits = [0, 5]
    >>> t1 = sqrt_trans()
    >>> t1.minor_breaks(major, limits)
    array([1.58113883, 2.54950976, 3.53553391])

    # Changing the regular `minor_breaks` method

    >>> t2 = sqrt_trans()
    >>> t2.minor_breaks = minor_breaks()
    >>> t2.minor_breaks(major, limits)
    array([0.5, 1.5, 2.5, 3.5, 4.5])

    More than 1 minor break

    >>> major = [1, 10]
    >>> limits = [1, 10]
    >>> t2.minor_breaks(major, limits, 4)
    array([2.8, 4.6, 6.4, 8.2])
    """

    trans: Trans
    n: int = 1

    def __call__(
        self,
        major: FloatArrayLike,
        limits: tuple[float, float] | None = None,
        n: int | None = None,
    ) -> NDArrayFloat:
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
        n : int
            Number of minor breaks between the major
            breaks. If **None**, then *self.n* is used.

        Returns
        -------
        out : array_like
            Minor breaks
        """
        if limits is None:
            limits = min_max(major)

        if n is None:
            n = self.n

        major = self._extend_breaks(major)
        major = self.trans.inverse(major)
        limits = self.trans.inverse(limits)
        minor = minor_breaks(n)(major, limits)
        return self.trans.transform(minor)

    def _extend_breaks(self, major: FloatArrayLike) -> FloatArrayLike:
        """
        Append 2 extra breaks at either end of major

        If breaks of transform space are non-equidistant,
        :func:`minor_breaks` add minor breaks beyond the first
        and last major breaks. The solutions is to extend those
        breaks (in transformed space) before the minor break call
        is made. How the breaks depends on the type of transform.
        """
        trans = self.trans
        trans = trans if isinstance(trans, type) else trans.__class__
        # so far we are only certain about this extending stuff
        # making sense for log transform
        is_log = trans.__name__.startswith("log")
        diff = np.diff(major)
        step = diff[0]
        if is_log and all(diff == step):
            major = np.hstack([major[0] - step, major, major[-1] + step])
        return major


@dataclass
class breaks_date:
    """
    Regularly spaced dates

    Parameters
    ----------
    n :
        Desired number of breaks.

    Examples
    --------
    >>> from datetime import datetime
    >>> limits = (datetime(2010, 1, 1), datetime(2026, 1, 1))

    Default breaks will be regularly spaced but the spacing
    is automatically determined

    >>> breaks = breaks_date(9)
    >>> [d.year for d in breaks(limits)]
    [2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024, 2026]
    """

    n: int = 5

    def __call__(
        self, limits: tuple[datetime, datetime] | tuple[date, date]
    ) -> Sequence[datetime]:
        """
        Compute breaks

        Parameters
        ----------
        limits : tuple
            Minimum and maximum :class:`datetime.datetime` values.

        Returns
        -------
        out : array_like
            Sequence of break points.
        """
        from mizani._datetime.breaks import by_n
        from mizani._datetime.utils import as_datetime

        if pd.isna(limits[0]) or pd.isna(limits[1]):
            return []

        if isinstance(limits[0], np.datetime64) and isinstance(
            limits[1], np.datetime64
        ):
            limits = limits[0].astype(object), limits[1].astype(object)

        limits = as_datetime(limits)
        return by_n(limits, self.n)


@dataclass
class breaks_date_width:
    """
    Regularly spaced dates by width

    Parameters
    ----------
    width : str
        The interval between the  breaks. A string of the form,
        "<number> <units>"`. The units are one of:

            microseconds
            milliseconds
            seconds
            minutes
            hours
            days
            weeks
            months
            years
            decades
            centuries

        or their singular forms. `secs` and `mins` or their singular forms
        are also recognised as abbreviations for seconds and minutes.

    offset : int | timedelta | str | Sequence[str] | relativedelta | None
        The breaks are set to start at some "nice" value but apply an
        offset you can shift them to a value you may prefer..

        - If an `int`, the units will be the same as the width.
        - If a `Sequence`, it is of the form
          `("[+-]<number> <units>", "[+-]<number> <units>", ...)`
          e.g. `("1 year", "2 months", ...)`.
        - If a `str`, it is of the form `"[+-]<number> <units>"`
          e.g. `"2 years"`.
        - If `None`, do not shift.

    Examples
    --------
    Breaks at 4 year intervals

    >>> limits = [datetime(2010, 1, 1), datetime(2025, 1, 1)]
    >>> breaks = breaks_date_width("4 years")
    >>> [d.year for d in breaks(limits)]
    [2010, 2014, 2018, 2022, 2026]
    >>> breaks = breaks_date_width("4 years", offset=1)
    >>> [d.year for d in breaks(limits)]
    [2011, 2015, 2019, 2023, 2027]
    """

    width: str
    offset: int | DatetimeOffset = None

    def __call__(
        self, limits: tuple[datetime, datetime] | tuple[date, date]
    ) -> Sequence[datetime]:
        """
        Compute breaks

        Parameters
        ----------
        limits :
            Minimum and maximum :class:`datetime.datetime` values.

        Returns
        -------
        out :
            Sequence of break points.
        """
        from mizani._datetime.breaks import by_width
        from mizani._datetime.utils import as_datetime

        if pd.isna(limits[0]) or pd.isna(limits[1]):
            return []

        if isinstance(limits[0], np.datetime64) and isinstance(
            limits[1], np.datetime64
        ):
            limits = limits[0].astype(object), limits[1].astype(object)

        limits = as_datetime(limits)
        return by_width(limits, self.width, self.offset)


@dataclass
class breaks_timedelta:
    """
    Timedelta breaks

    Returns
    -------
    out : callable ``f(limits)``
        A function that takes a sequence of two
        :class:`datetime.timedelta` values and returns
        a sequence of break points.

    Examples
    --------
    >>> from datetime import timedelta
    >>> breaks = breaks_timedelta()
    >>> limits = (timedelta(days=0), timedelta(days=345))
    >>> major = breaks(limits)
    >>> [b.days for b in major]
    [0, 70, 140, 210, 280, 350]
    """

    n: int = 5

    def __call__(
        self, limits: tuple[Timedelta, Timedelta]
    ) -> TimedeltaArrayLike:
        """
        Compute breaks

        Parameters
        ----------
        limits : tuple
            Minimum and maximum :class:`datetime.timedelta` values.

        Returns
        -------
        out : array_like
            Sequence of break points.
        """
        from mizani._timedelta.breaks import by_n

        return by_n(limits, self.n)


@dataclass
class breaks_timedelta_width:
    """
    Regularly spaced timedeltas by width

    Parameters
    ----------
    width : str
        The interval between the  breaks. A string of the form,
        "<number> <units>"`. The units are one of:

            microseconds
            milliseconds
            seconds
            minutes
            hours
            days
            weeks

    offset :
        Use this to shift the calculated breaks so that they start at
        a value you may prefer.

        - If an `int`, the units will be the same as the width.
        - If a `Sequence`, it is of the form
          `("[+-]<number> <units>", "[+-]<number> <units>", ...)`
          e.g. `("2 days", "12 hours", ...)`.
        - If a `str`, it is of the form `"[+-]<number> <units>"`
          e.g. `"4 hours"`
        - If `None`, do not shift.
    """

    width: str
    offset: int | TimedeltaOffset = None

    def __call__(
        self, limits: tuple[Timedelta, Timedelta]
    ) -> TimedeltaArrayLike:
        """
        Compute breaks

        Parameters
        ----------
        limits :
            Minimum and maximum :class:`datetime.timedelta` values.

        Returns
        -------
        out :
            Sequence of break points.
        """
        from mizani._timedelta.breaks import by_width

        return by_width(limits, self.width, self.offset)


@dataclass
class breaks_extended:
    """
    An extension of Wilkinson's tick position algorithm

    Parameters
    ----------
    n : int
        Desired number of breaks
    Q : list
        List of nice numbers
    only_inside : bool
        If ``True``, then all the breaks will be within the given
        range.
    w : list
        Weights applied to the four optimization components
        (simplicity, coverage, density, and legibility). They
        should add up to 1.

    Examples
    --------
    >>> limits = (0, 9)
    >>> breaks_extended()(limits)
    array([  0. ,   2.5,   5. ,   7.5,  10. ])
    >>> breaks_extended(n=6)(limits)
    array([  0.,   2.,   4.,   6.,   8.,  10.])

    References
    ----------
    - Talbot, J., Lin, S., Hanrahan, P. (2010) An Extension of
      Wilkinson's Algorithm for Positioning Tick Labels on Axes,
      InfoVis 2010.

    Additional Credit to Justin Talbot on whose code this
    implementation is almost entirely based.
    """

    n: int = 5
    Q: Sequence[float] = (1, 5, 2, 2.5, 4, 3)
    only_inside: bool = False
    w: Sequence[float] = (0.25, 0.2, 0.5, 0.05)

    def __post_init__(self):
        # Used for lookups during the computations
        self.Q_index = {q: i for i, q in enumerate(self.Q)}

    def coverage(
        self, dmin: float, dmax: float, lmin: float, lmax: float
    ) -> float:
        p1 = (dmax - lmax) ** 2
        p2 = (dmin - lmin) ** 2
        p3 = (0.1 * (dmax - dmin)) ** 2
        return 1 - 0.5 * (p1 + p2) / p3

    def coverage_max(self, dmin: float, dmax: float, span: float) -> float:
        range = dmax - dmin
        if span > range:
            half = (span - range) / 2.0
            return 1 - (half**2) / (0.1 * range) ** 2
        else:
            return 1

    def density(
        self, k: float, dmin: float, dmax: float, lmin: float, lmax: float
    ) -> float:
        r = (k - 1.0) / (lmax - lmin)
        rt = (self.n - 1) / (max(lmax, dmax) - min(lmin, dmin))
        return 2 - max(r / rt, rt / r)

    def density_max(self, k: float) -> float:
        if k >= self.n:
            return 2 - (k - 1.0) / (self.n - 1.0)
        else:
            return 1

    def simplicity(
        self, q: float, j: float, lmin: float, lmax: float, lstep: float
    ) -> float:
        eps = 1e-10
        n = len(self.Q)
        i = self.Q_index[q] + 1

        if (
            (lmin % lstep < eps or (lstep - lmin % lstep) < eps)
            and lmin <= 0
            and lmax >= 0
        ):
            v = 1
        else:
            v = 0
        return (n - i) / (n - 1.0) + v - j

    def simplicity_max(self, q: float, j: float) -> float:
        n = len(self.Q)
        i = self.Q_index[q] + 1
        v = 1
        return (n - i) / (n - 1.0) + v - j

    def legibility(self, lmin: float, lmax: float, lstep: float) -> float:
        # Legibility depends on fontsize, rotation, overlap ... i.e.
        # it requires drawing or simulating drawn breaks then calculating
        # a score. Return 1 ignores all that.
        return 1

    def __call__(self, limits: tuple[float, float]) -> NDArrayFloat:
        """
        Calculate the breaks

        Parameters
        ----------
        limits : array
            Minimum and maximum values.

        Returns
        -------
        out : array_like
            Sequence of break points.
        """
        Q = self.Q
        w = self.w
        only_inside = self.only_inside
        simplicity_max = self.simplicity_max
        density_max = self.density_max
        coverage_max = self.coverage_max
        simplicity = self.simplicity
        coverage = self.coverage
        density = self.density
        legibility = self.legibility
        log10 = np.log10
        ceil = np.ceil
        floor = np.floor
        # casting prevents the typechecker from mixing
        # float & np.float32
        dmin, dmax = float(limits[0]), float(limits[1])

        if dmin > dmax:
            dmin, dmax = dmax, dmin
        elif dmin == dmax:
            return np.array([dmin])

        best_score = -2.0
        best = (0, 0, 0, 0, 0)  # Gives Empty breaks
        j = 1.0

        while j < float("inf"):
            for q in Q:
                sm = simplicity_max(q, j)

                if w[0] * sm + w[1] + w[2] + w[3] < best_score:
                    j = float("inf")
                    break

                k = 2.0
                while k < float("inf"):
                    dm = density_max(k)

                    if w[0] * sm + w[1] + w[2] * dm + w[3] < best_score:
                        break

                    delta = (dmax - dmin) / (k + 1) / j / q
                    z: float = ceil(log10(delta))

                    while z < float("inf"):
                        step = j * q * (10**z)
                        cm = coverage_max(dmin, dmax, step * (k - 1))

                        if (
                            w[0] * sm + w[1] * cm + w[2] * dm + w[3]
                            < best_score
                        ):
                            break

                        min_start = int(floor(dmax / step) * j - (k - 1) * j)
                        max_start = int(ceil(dmin / step) * j)

                        if min_start > max_start:
                            z = z + 1
                            break

                        for start in range(min_start, max_start + 1):
                            lmin = start * (step / j)
                            lmax = lmin + step * (k - 1)
                            lstep = step

                            s = simplicity(q, j, lmin, lmax, lstep)
                            c = coverage(dmin, dmax, lmin, lmax)
                            d = density(k, dmin, dmax, lmin, lmax)
                            l = legibility(lmin, lmax, lstep)

                            score = w[0] * s + w[1] * c + w[2] * d + w[3] * l

                            if score > best_score and (
                                not only_inside
                                or (lmin >= dmin and lmax <= dmax)
                            ):
                                best_score = score
                                best = (lmin, lmax, lstep, q, k)
                        z = z + 1
                    k = k + 1
            j = j + 1

        locs = best[0] + np.arange(best[4]) * best[2]
        return locs


class breaks_symlog:
    """
    Breaks for the Symmetric Logarithm Transform


    Examples
    --------
    >>> limits = (-100, 100)
    >>> breaks_symlog()(limits)
    array([-100,  -10,    0,   10,  100])
    """

    def __call__(self, limits: tuple[float, float]) -> NDArrayFloat:
        def _signed_log10(x):
            return np.round(np.sign(x) * np.log10(np.sign(x) * x)).astype(int)

        l, h = _signed_log10(limits)
        exps = np.arange(l, h + 1, 1)
        return np.sign(exps) * (10 ** np.abs(exps))


@dataclass
class breaks_width:
    """
    Regularly spaced dates by width

    Parameters
    ----------
    width :
        The interval between the breaks.

    offset :
        Shift the calculated breaks by this much.

    Examples
    --------
    Breaks at 4 year intervals

    >>> limits = [3, 14]
    >>> breaks = breaks_width(width=4)
    >>> breaks(limits)
    array([ 0,  4,  8, 12, 16])
    """

    width: float
    offset: float | None = None

    def __call__(self, limits: tuple[float, float]) -> NDArrayFloat:
        offset = 0 if self.offset is None else self.offset
        start = round_any(limits[0], self.width, np.floor) + offset
        end = round_any(limits[1], self.width, np.ceil) + self.width
        dtype = (
            int
            if isinstance(self.width, int) and isinstance(offset, int)
            else float
        )
        return np.arange(start, end, self.width, dtype=dtype)


# Deprecated
log_breaks = breaks_log
trans_minor_breaks = minor_breaks_trans
date_breaks = breaks_width
breaks_timedelta = breaks_timedelta
extended_breaks = breaks_extended
