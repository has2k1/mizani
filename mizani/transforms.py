"""
*"The Grammar of Graphics (2005)"* by **Wilkinson**, **Anand** and
**Grossman** describes three types of transformations.

* *Variable transformations* - Used to make statistical operations on
  variables appropriate and meaningful. They are also used to new
  variables.
* *Scale transformations* - Used to make statistical objects displayed
  on dimensions appropriate and meaningful.
* *Coordinate transformations* - Used to manipulate the geometry of
  graphics to help perceive relationships and find meaningful structures
  for representing variations.

`Variable` and `scale` transformations are similar in-that they lead to
plotted objects that are indistinguishable. Typically, *variable*
transformation is done outside the graphics system and so the system
cannot provide transformation specific guides & decorations for the
plot. The :class:`trans` is aimed at being useful for *scale* and
*coordinate* transformations.
"""

from __future__ import annotations

import sys
import typing
from abc import ABC, abstractmethod
from datetime import MAXYEAR, MINYEAR, datetime, timedelta
from types import MethodType
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ._core.dates import datetime_to_num, num_to_datetime
from .breaks import (
    breaks_date,
    breaks_extended,
    breaks_log,
    breaks_symlog,
    breaks_timedelta,
    minor_breaks,
    minor_breaks_trans,
)
from .labels import (
    label_date,
    label_log,
    label_number,
    label_timedelta,
)
from .utils import identity

if typing.TYPE_CHECKING:
    from typing import Any, Callable, Optional, Sequence, Type

    from mizani.typing import (
        BreaksFunction,
        DatetimeArrayLike,
        DomainType,
        FloatArrayLike,
        FormatFunction,
        InverseFunction,
        MinorBreaksFunction,
        NDArrayAny,
        NDArrayDatetime,
        NDArrayFloat,
        NDArrayTimedelta,
        TFloatArrayLike,
        TimedeltaSeries,
        TransformFunction,
        TupleFloat2,
    )

__all__ = [
    "asn_trans",
    "atanh_trans",
    "boxcox_trans",
    "modulus_trans",
    "datetime_trans",
    "exp_trans",
    "identity_trans",
    "log10_trans",
    "log1p_trans",
    "log2_trans",
    "log_trans",
    "logit_trans",
    "probability_trans",
    "probit_trans",
    "reverse_trans",
    "sqrt_trans",
    "symlog_trans",
    "timedelta_trans",
    "pd_timedelta_trans",
    "pseudo_log_trans",
    "reciprocal_trans",
    "trans",
    "trans_new",
    "gettrans",
]

UTC = ZoneInfo("UTC")


class trans(ABC):
    """
    Base class for all transforms

    This class is used to transform data and also tell the
    x and y axes how to create and label the tick locations.

    The key methods to override are :meth:`trans.transform`
    and :meth:`trans.inverse`. Alternately, you can quickly
    create a transform class using the :func:`trans_new`
    function.

    Parameters
    ----------
    kwargs : dict
        Attributes of the class to set/override

    """

    #: Whether the transformation over the whole domain is linear.
    #: e.g. `2x` is linear while `1/x` and `log(x)` are not.
    transform_is_linear: bool = False

    domain: DomainType = (-np.inf, np.inf)

    #: Callable to calculate breaks
    breaks_: BreaksFunction = breaks_extended(n=5)

    #: Function to format breaks
    format: FormatFunction = staticmethod(label_number())

    def __init__(self, **kwargs: Any):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise AttributeError(f"Unknown Parameter: {k}")

    @property
    def domain_is_numerical(self) -> bool:
        """
        Return True if transformation acts on numerical data.
        e.g. int, float, and imag are numerical but datetime
        is not.

        """
        return isinstance(self.domain[0], (int, float, np.number))

    def minor_breaks(
        self,
        major: FloatArrayLike,
        limits: Optional[TupleFloat2] = None,
        n: Optional[int] = None,
    ) -> NDArrayFloat:
        """
        Calculate minor_breaks
        """
        n = 1 if n is None else n

        # minor_breaks_trans undoes the transformation and
        # then calculates the breaks. If the domain/dataspace
        # numerical, the calculation will fail.
        if self.transform_is_linear or not self.domain_is_numerical:
            func = minor_breaks(n=n)
        else:
            func = minor_breaks_trans(self, n=n)
        return func(major, limits, n)

    # Use type variables for trans.transform and trans.inverse
    # to help upstream packages avoid type mismatches. e.g.
    # transform(tuple[float, float]) -> tuple[float, float]
    @staticmethod
    @abstractmethod
    def transform(x: TFloatArrayLike) -> TFloatArrayLike:
        """
        Transform of x
        """
        ...

    @staticmethod
    @abstractmethod
    def inverse(x: TFloatArrayLike) -> TFloatArrayLike:
        """
        Inverse of x
        """
        ...

    def breaks(self, limits: tuple[Any, Any]) -> NDArrayAny:
        """
        Calculate breaks in data space and return them
        in transformed space.

        Expects limits to be in *transform space*, this
        is the same space as that where the domain is
        specified.

        This method wraps around :meth:`breaks_` to ensure
        that the calculated breaks are within the domain
        the transform. This is helpful in cases where an
        aesthetic requests breaks with limits expanded for
        some padding, yet the expansion goes beyond the
        domain of the transform. e.g for a probability
        transform the breaks will be in the domain
        ``[0, 1]`` despite any outward limits.

        Parameters
        ----------
        limits : tuple
            The scale limits. Size 2.

        Returns
        -------
        out : array_like
            Major breaks
        """
        # clip the breaks to the domain,
        # e.g. probabilities will be in [0, 1] domain
        limits = (
            max(self.domain[0], limits[0]),
            min(self.domain[1], limits[1]),
        )
        breaks = np.asarray(self.breaks_(limits))

        # Some methods (e.g. breaks_extended) that
        # calculate breaks take the limits as guide posts and
        # not hard limits.
        breaks = breaks.compress(
            (breaks >= self.domain[0]) & (breaks <= self.domain[1])
        )
        return breaks


def trans_new(
    name: str,
    transform: TransformFunction,
    inverse: InverseFunction,
    breaks: Optional[BreaksFunction] = None,
    minor_breaks: Optional[MinorBreaksFunction] = None,
    _format: Optional[FormatFunction] = None,
    domain=(-np.inf, np.inf),
    doc: str = "",
    **kwargs,
) -> trans:
    """
    Create a transformation class object

    Parameters
    ----------
    name : str
        Name of the transformation
    transform : callable ``f(x)``
        A function (preferably a `ufunc`) that computes
        the transformation.
    inverse : callable ``f(x)``
        A function (preferably a `ufunc`) that computes
        the inverse of the transformation.
    breaks : callable ``f(limits)``
        Function to compute the breaks for this transform.
        If None, then a default good enough for a linear
        domain is used.
    minor_breaks : callable ``f(major, limits)``
        Function to compute the minor breaks for this
        transform. If None, then a default good enough for
        a linear domain is used.
    _format : callable ``f(breaks)``
        Function to format the generated breaks.
    domain : array_like
        Domain over which the transformation is valid.
        It should be of length 2.
    doc : str
        Docstring for the class.
    **kwargs : dict
        Attributes of the transform, e.g if base is passed
        in kwargs, then `t.base` would be a valied attribute.

    Returns
    -------
    out : trans
        Transform class
    """

    def _get(func):
        if isinstance(func, (classmethod, staticmethod, MethodType)):
            return func
        else:
            return staticmethod(func)

    klass_name = "{}_trans".format(name)

    d = {
        "transform": _get(transform),
        "inverse": _get(inverse),
        "domain": domain,
        "__doc__": doc,
        **kwargs,
    }

    if breaks:
        d["breaks_"] = _get(breaks)

    if minor_breaks:
        d["minor_breaks"] = _get(minor_breaks)

    if _format:
        d["format"] = _get(_format)

    return type(klass_name, (trans,), d)  # type: ignore


def log_trans(base: Optional[float] = None, **kwargs: Any) -> trans:
    """
    Create a log transform class for *base*

    Parameters
    ----------
    base : float
        Base for the logarithm. If None, then
        the natural log is used.
    kwargs : dict
        Keyword arguments passed onto
        :func:`trans_new`. Should not include
        the `transform` or `inverse`.

    Returns
    -------
    out : type
        Log transform class
    """
    # transform function
    if base is None:
        name = "log"
        base = np.exp(1)
        transform = np.log  # type: ignore
    elif base == 10:
        name = "log10"
        transform = np.log10  # type: ignore
    elif base == 2:
        name = "log2"
        transform = np.log2  # type: ignore
    else:
        name = "log{}".format(base)

        def transform(x: FloatArrayLike) -> NDArrayFloat:
            return np.log(x) / np.log(base)

    # inverse function
    def inverse(x):
        return np.power(base, x)  # type: ignore

    if "domain" not in kwargs:
        kwargs["domain"] = (sys.float_info.min, np.inf)

    if "breaks" not in kwargs:
        kwargs["breaks"] = breaks_log(base=base)  # type: ignore

    kwargs["base"] = base
    kwargs["_format"] = label_log(base)  # type: ignore

    _trans = trans_new(name, transform, inverse, **kwargs)

    if "minor_breaks" not in kwargs:
        n = int(base) - 2  # type: ignore
        _trans.minor_breaks = minor_breaks_trans(_trans, n=n)

    return _trans


log10_trans = log_trans(10, doc="Log 10 Transformation")
log2_trans = log_trans(2, doc="Log 2 Transformation")


def exp_trans(base: Optional[float] = None, **kwargs: Any):
    """
    Create a exponential transform class for *base*

    This is inverse of the log transform.

    Parameters
    ----------
    base : float
        Base of the logarithm
    kwargs : dict
        Keyword arguments passed onto
        :func:`trans_new`. Should not include
        the `transform` or `inverse`.

    Returns
    -------
    out : type
        Exponential transform class
    """
    # default to e
    if base is None:
        name = "power_e"
        base = np.exp(1)
    else:
        name = "power_{}".format(base)

    # transform function
    def transform(x):
        return np.power(base, x)  # type: ignore

    # inverse function
    def inverse(x):
        return np.log(x) / np.log(base)  # type: ignore

    kwargs["base"] = base
    return trans_new(name, transform, inverse, **kwargs)


class log1p_trans(trans):
    """
    Log plus one Transformation
    """

    transform = staticmethod(np.log1p)  # type: ignore
    inverse = staticmethod(np.expm1)  # type: ignore


class identity_trans(trans):
    """
    Identity Transformation

    Examples
    --------
    The default trans returns one minor break between every pair
    of major break

    >>> major = [0, 1, 2]
    >>> t = identity_trans()
    >>> t.minor_breaks(major)
    array([0.5, 1.5])

    Create a trans that returns 4 minor breaks

    >>> t = identity_trans(minor_breaks=minor_breaks(4))
    >>> t.minor_breaks(major)
    array([0.2, 0.4, 0.6, 0.8, 1.2, 1.4, 1.6, 1.8])
    """

    transform_is_linear = True
    transform = staticmethod(identity)  # type: ignore
    inverse = staticmethod(identity)  # type: ignore


class reverse_trans(trans):
    """
    Reverse Transformation
    """

    transform_is_linear = True
    transform = staticmethod(np.negative)  # type: ignore
    inverse = staticmethod(np.negative)  # type: ignore


class sqrt_trans(trans):
    """
    Square-root Transformation
    """

    transform = staticmethod(np.sqrt)  # type: ignore
    inverse = staticmethod(np.square)  # type: ignore
    domain = (0, np.inf)


class asn_trans(trans):
    """
    Arc-sin square-root Transformation
    """

    @staticmethod
    def transform(x: FloatArrayLike) -> NDArrayFloat:
        return 2 * np.arcsin(np.sqrt(x))  # type: ignore

    @staticmethod
    def inverse(x: FloatArrayLike) -> NDArrayFloat:
        x = np.asarray(x)
        return np.sin(x / 2) ** 2  # type: ignore


class atanh_trans(trans):
    """
    Arc-tangent Transformation
    """

    transform = staticmethod(np.arctanh)  # type: ignore
    inverse = staticmethod(np.tanh)  # type: ignore


def boxcox_trans(p, offset=0, **kwargs):
    r"""
    Boxcox Transformation

    The Box-Cox transformation is a flexible transformation, often
    used to transform data towards normality.

    The Box-Cox power transformation (type 1) requires strictly positive
    values and takes the following form for :math:`y \gt 0`:

    .. math::

        y^{(\lambda)} = \frac{y^\lambda - 1}{\lambda}

    When :math:`y = 0`, the natural log transform is used.

    Parameters
    ----------
    p : float
        Transformation exponent :math:`\lambda`.
    offset : int
        Constant offset. 0 for Box-Cox type 1, otherwise any
        non-negative constant (Box-Cox type 2).
        The default is 0. :func:`~mizani.transforms.modulus_trans`
        sets the default to 1.
    kwargs : dict
        Keyword arguments passed onto :func:`trans_new`. Should not
        include the `transform` or `inverse`.

    References
    ----------
    - Box, G. E., & Cox, D. R. (1964). An analysis of transformations.
      Journal of the Royal Statistical Society. Series B (Methodological),
      211-252. `<https://www.jstor.org/stable/2984418>`_
    - John, J. A., & Draper, N. R. (1980). An alternative family of
      transformations. Applied Statistics, 190-197.
      `<http://www.jstor.org/stable/2986305>`_

    See Also
    --------
    :func:`~mizani.transforms.modulus_trans`

    """

    def transform(x: FloatArrayLike) -> NDArrayFloat:
        x = np.asarray(x)
        if np.any((x + offset) < 0):
            raise ValueError(
                "boxcox_trans must be given only positive values. "
                "Consider using modulus_trans instead?"
            )
        if np.abs(p) < 1e-7:
            return np.log(x + offset)
        else:
            return ((x + offset) ** p - 1) / p

    def inverse(x: FloatArrayLike) -> NDArrayFloat:
        x = np.asarray(x)
        if np.abs(p) < 1e-7:
            return np.exp(x) - offset  # type: ignore
        else:
            return (x * p + 1) ** (1 / p) - offset

    kwargs["p"] = p
    kwargs["offset"] = offset
    kwargs["name"] = kwargs.get("name", "pow_{}".format(p))
    kwargs["transform"] = transform
    kwargs["inverse"] = inverse
    return trans_new(**kwargs)


def modulus_trans(p, offset=1, **kwargs):
    r"""
    Modulus Transformation

    The modulus transformation generalises Box-Cox to work with
    both positive and negative values.

    When :math:`y \neq 0`

    .. math::

        y^{(\lambda)} = sign(y) * \frac{(|y| + 1)^\lambda - 1}{\lambda}

    and when :math:`y = 0`

    .. math::

        y^{(\lambda)} =  sign(y) * \ln{(|y| + 1)}

    Parameters
    ----------
    p : float
        Transformation exponent :math:`\lambda`.
    offset : int
        Constant offset. 0 for Box-Cox type 1, otherwise any
        non-negative constant (Box-Cox type 2).
        The default is 1. :func:`~mizani.transforms.boxcox_trans`
        sets the default to 0.
    kwargs : dict
        Keyword arguments passed onto :func:`trans_new`.
        Should not include the `transform` or `inverse`.

    References
    ----------
    - Box, G. E., & Cox, D. R. (1964). An analysis of transformations.
      Journal of the Royal Statistical Society. Series B (Methodological),
      211-252. `<https://www.jstor.org/stable/2984418>`_
    - John, J. A., & Draper, N. R. (1980). An alternative family of
      transformations. Applied Statistics, 190-197.
      `<http://www.jstor.org/stable/2986305>`_

    See Also
    --------
    :func:`~mizani.transforms.boxcox_trans`
    """
    if np.abs(p) < 1e-7:

        def transform(x: FloatArrayLike) -> NDArrayFloat:
            x = np.asarray(x)
            return np.sign(x) * np.log(np.abs(x) + offset)  # type: ignore

        def inverse(x: FloatArrayLike) -> NDArrayFloat:
            x = np.asarray(x)
            return np.sign(x) * (np.exp(np.abs(x)) - offset)  # type: ignore

    else:

        def transform(x: FloatArrayLike) -> NDArrayFloat:
            x = np.asarray(x)
            return np.sign(x) * ((np.abs(x) + offset) ** p - 1) / p

        def inverse(x: FloatArrayLike) -> NDArrayFloat:
            x = np.asarray(x)
            return np.sign(x) * ((np.abs(x) * p + 1) ** (1 / p) - offset)

    kwargs["p"] = p
    kwargs["offset"] = offset
    kwargs["name"] = kwargs.get("name", "mt_pow_{}".format(p))
    kwargs["transform"] = transform
    kwargs["inverse"] = inverse
    return trans_new(**kwargs)


def probability_trans(distribution: str, *args, **kwargs) -> trans:
    """
    Probability Transformation

    Parameters
    ----------
    distribution : str
        Name of the distribution. Valid distributions are
        listed at :mod:`scipy.stats`. Any of the continuous
        or discrete distributions.
    args : tuple
        Arguments passed to the distribution functions.
    kwargs : dict
        Keyword arguments passed to the distribution functions.

    Notes
    -----
    Make sure that the distribution is a good enough
    approximation for the data. When this is not the case,
    computations may run into errors. Absence of any errors
    does not imply that the distribution fits the data.
    """
    import scipy.stats as stats

    cdists = {k for k in dir(stats) if hasattr(getattr(stats, k), "cdf")}
    if distribution not in cdists:
        raise ValueError(f"Unknown distribution '{distribution}'")

    try:
        doc = kwargs.pop("_doc")
    except KeyError:
        doc = ""

    try:
        name = kwargs.pop("_name")
    except KeyError:
        name = "prob_{}".format(distribution)

    def transform(x: FloatArrayLike) -> NDArrayFloat:
        return getattr(stats, distribution).cdf(x, *args, **kwargs)

    def inverse(x: FloatArrayLike) -> NDArrayFloat:
        return getattr(stats, distribution).ppf(x, *args, **kwargs)

    return trans_new(name, transform, inverse, domain=(0, 1), doc=doc)


logit_trans = probability_trans(
    "logistic", _name="logit", _doc="Logit Transformation"
)
probit_trans = probability_trans(
    "norm", _name="norm", _doc="Probit Transformation"
)


class datetime_trans(trans):
    """
    Datetime Transformation

    Parameters
    ----------
    tz : str | ZoneInfo
        Timezone information

    Examples
    --------
    >>> from zoneinfo import ZoneInfo
    >>> UTC = ZoneInfo("UTC")
    >>> EST = ZoneInfo("EST")
    >>> t = datetime_trans(EST)
    >>> x = [datetime(2022, 1, 20, tzinfo=UTC)]
    >>> x2 = t.inverse(t.transform(x))
    >>> list(x) == list(x2)
    True
    >>> x[0].tzinfo == x2[0].tzinfo
    False
    >>> x[0].tzinfo.key
    'UTC'
    >>> x2[0].tzinfo.key
    'EST'
    """

    domain = (
        datetime(MINYEAR, 1, 1, tzinfo=UTC),
        datetime(MAXYEAR, 12, 31, tzinfo=UTC),
    )
    breaks_ = staticmethod(breaks_date())
    format = staticmethod(label_date())
    tz = None

    def __init__(self, tz=None, **kwargs):
        if isinstance(tz, str):
            tz = ZoneInfo(tz)

        super().__init__(**kwargs)
        self.tz = tz

    def transform(self, x: DatetimeArrayLike) -> NDArrayFloat:  # pyright: ignore[reportIncompatibleMethodOverride]
        """
        Transform from date to a numerical format
        """
        if not len(x):
            return np.array([])

        x0 = next(iter(x))
        try:
            tz = x0.tzinfo
        except AttributeError:
            tz = None

        if tz and self.tz is None:
            self.tz = tz

        return datetime_to_num(x)  # type: ignore

    def inverse(self, x: FloatArrayLike) -> NDArrayDatetime:  # pyright: ignore[reportIncompatibleMethodOverride]
        """
        Transform to date from numerical format
        """
        return num_to_datetime(x, tz=self.tz)

    @property
    def tzinfo(self):
        """
        Alias of `tz`
        """
        return self.tz


class timedelta_trans(trans):
    """
    Timedelta Transformation
    """

    domain = (timedelta.min, timedelta.max)
    breaks_ = staticmethod(breaks_timedelta())
    format = staticmethod(label_timedelta())

    @staticmethod
    def transform(x: NDArrayTimedelta | Sequence[timedelta]) -> NDArrayFloat:  # pyright: ignore[reportIncompatibleMethodOverride]
        """
        Transform from Timeddelta to numerical format
        """
        # microseconds
        return np.array([_x.total_seconds() * 10**6 for _x in x])

    @staticmethod
    def inverse(x: FloatArrayLike) -> NDArrayTimedelta:
        """
        Transform to Timedelta from numerical format
        """
        return np.array([timedelta(microseconds=i) for i in x])


class pd_timedelta_trans(trans):
    """
    Pandas timedelta Transformation
    """

    domain = (pd.Timedelta.min, pd.Timedelta.max)
    breaks_ = staticmethod(breaks_timedelta())
    format = staticmethod(label_timedelta())

    @staticmethod
    def transform(x: TimedeltaSeries) -> NDArrayFloat:  # pyright: ignore[reportIncompatibleMethodOverride]
        """
        Transform from Timeddelta to numerical format
        """
        # nanoseconds
        return np.array([_x.value for _x in x])

    @staticmethod
    def inverse(x: FloatArrayLike) -> NDArrayTimedelta:
        """
        Transform to Timedelta from numerical format
        """
        return np.array([pd.Timedelta(int(i)) for i in x])


class reciprocal_trans(trans):
    """
    Reciprocal Transformation
    """

    @staticmethod
    def transform(x: FloatArrayLike) -> NDArrayFloat:
        return 1 / np.asarray(x)

    @staticmethod
    def inverse(x: FloatArrayLike) -> NDArrayFloat:
        return 1 / np.asarray(x)


class pseudo_log_trans(trans):
    """
    Pseudo-log transformation

    A transformation mapping numbers to a signed logarithmic
    scale with a smooth transition to linear scale around 0.

    Parameters
    ----------
    sigma : float
        Scaling factor for the linear part.
    base : int
        Approximate logarithm used. If None, then
        the natural log is used.
    kwargs : dict
        Keyword arguments passed onto
        :func:`trans_new`. Should not include
        the `transform` or `inverse`.
    """

    def __init__(self, sigma=1, base=None, **kwargs):
        if base is None:
            base = np.exp(1)

        self.sigma = sigma
        self.base = base
        super().__init__(**kwargs)

    def transform(self, x: FloatArrayLike) -> NDArrayFloat:  # pyright: ignore[reportIncompatibleMethodOverride]
        x = np.asarray(x)
        return np.arcsinh(x / (2 * self.sigma)) / np.log(self.base)

    def inverse(self, x: FloatArrayLike) -> NDArrayFloat:  # pyright: ignore[reportIncompatibleMethodOverride]
        x = np.asarray(x)
        return 2 * self.sigma * np.sinh(x * np.log(self.base))

    def minor_breaks(
        self,
        major: FloatArrayLike,
        limits: Optional[TupleFloat2] = None,
        n: Optional[int] = None,
    ) -> NDArrayFloat:
        n = int(self.base) - 2 if n is None else n
        return super().minor_breaks(major, limits, n)


class symlog_trans(trans):
    """
    Symmetric Log Transformation

    They symmetric logarithmic transformation is defined as

    ::

        f(x) = log(x+1) for x >= 0
               -log(-x+1) for x < 0

    It can be useful for data that has a wide range of both positive
    and negative values (including zero).
    """

    breaks_: BreaksFunction = breaks_symlog()

    @staticmethod
    def transform(x: FloatArrayLike) -> NDArrayFloat:
        return np.sign(x) * np.log1p(np.abs(x))  # type: ignore

    @staticmethod
    def inverse(x: FloatArrayLike) -> NDArrayFloat:
        return np.sign(x) * (np.exp(np.abs(x)) - 1)  # type: ignore


def gettrans(t: str | Callable[[], Type[trans]] | Type[trans] | trans):
    """
    Return a trans object

    Parameters
    ----------
    t : str | callable | type | trans
        name of transformation function

    Returns
    -------
    out : trans
    """
    obj = t
    # Make sure trans object is instantiated
    if isinstance(obj, str):
        name = "{}_trans".format(obj)
        obj = globals()[name]()
    if callable(obj):
        obj = obj()
    if isinstance(obj, type):
        obj = obj()

    if not isinstance(obj, trans):
        raise ValueError("Could not get transform object.")

    return obj
