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
from abc import ABC, abstractmethod
from dataclasses import KW_ONLY, dataclass, field
from datetime import MAXYEAR, MINYEAR, datetime, timedelta, tzinfo
from types import MethodType
from typing import TYPE_CHECKING
from warnings import warn
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ._core.dates import (
    datetime_to_num,
    num_to_datetime,
    num_to_timedelta,
    timedelta_to_num,
)
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

if TYPE_CHECKING:
    from typing import Any, Sequence, Type

    from mizani.typing import (
        BreaksFunction,
        DatetimeArrayLike,
        DomainType,
        FloatArrayLike,
        FormatFunction,
        InverseFunction,
        MinorBreaksFunction,
        NDArrayDatetime,
        NDArrayFloat,
        TFloatArrayLike,
        TimedeltaArrayLike,
        TransformFunction,
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
    "gettrans",
]

UTC = ZoneInfo("UTC")
REGISTRY: dict[str, Type[trans]] = {}


@dataclass(kw_only=True)
class trans(ABC):
    domain: DomainType = (-np.inf, np.inf)

    transform_is_linear: bool = False
    """
    Whether the transformation over the whole domain is linear.
    e.g. `2x` is linear while `1/x` and `log(x)` are not.
    """

    breaks_func: BreaksFunction = field(default_factory=breaks_extended)
    "Callable to calculate breaks"

    format_func: FormatFunction = field(default_factory=label_number)
    "Function to format breaks"

    minor_breaks_func: MinorBreaksFunction | None = None
    "Callable to calculate minor breaks"

    def __init_subclass__(cls, *args, **kwargs):
        # Register all subclasses
        super().__init_subclass__(*args, **kwargs)
        REGISTRY[cls.__name__] = cls

    # Use type variables for trans.transform and trans.inverse
    # to help upstream packages avoid type mismatches. e.g.
    # transform(tuple[float, float]) -> tuple[float, float]
    @abstractmethod
    def transform(self, x: TFloatArrayLike) -> TFloatArrayLike:
        """
        Transform of x
        """
        ...

    @abstractmethod
    def inverse(self, x: TFloatArrayLike) -> TFloatArrayLike:
        """
        Inverse of x
        """
        ...

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
        limits: tuple[float, float] | None = None,
        n: int | None = None,
    ) -> NDArrayFloat:
        """
        Calculate minor_breaks
        """
        if self.minor_breaks_func is not None:
            return self.minor_breaks_func(major, limits, n)

        n = 1 if n is None else n

        # minor_breaks_trans undoes the transformation and
        # then calculates the breaks. If the domain/dataspace
        # numerical, the calculation will fail.
        if self.transform_is_linear or not self.domain_is_numerical:
            func = minor_breaks(n=n)
        else:
            func = minor_breaks_trans(self, n=n)
        return func(major, limits, n)

    def breaks(self, limits: DomainType) -> NDArrayFloat:
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
        breaks = np.asarray(self.breaks_func(limits))

        # Some methods (e.g. breaks_extended) that
        # calculate breaks take the limits as guide posts and
        # not hard limits.
        breaks = breaks.compress(
            (breaks >= self.domain[0]) & (breaks <= self.domain[1])
        )
        return breaks

    def format(self, x: Any) -> Sequence[str]:
        """
        Format breaks

        When subclassing, you can override this function, or you can
        just define `format_func`.
        """
        return self.format_func(x)

    def diff_type_to_num(self, x: Any) -> FloatArrayLike:
        """
        Convert the difference between two points in the domain to a numeric

        This function is necessary for some arithmetic operations in the
        transform space of a domain when the difference in between any two
        points in that domain is not numeric.

        For example for a domain of datetime value types, the difference on
        the domain is of type timedelta. In this case this function should
        expect timedeltas and convert them to float values that compatible
        (same units) as the transform value of datetimes.

        Parameters
        ----------
        x :
            Differences
        """
        return x


def trans_new(
    name: str,
    transform: TransformFunction,
    inverse: InverseFunction,
    breaks_func: BreaksFunction | None = None,
    minor_breaks_func: MinorBreaksFunction | None = None,
    format_func: FormatFunction | None = None,
    domain: DomainType = (-np.inf, np.inf),
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
    warn(
        "This function has been deprecated and will be removed in a future "
        "version. You should create transforms explicitly using the class "
        "syntax.",
        FutureWarning,
    )

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

    if breaks_func:
        d["breaks_func"] = _get(breaks_func)

    if minor_breaks:
        d["minor_breaks_func"] = _get(minor_breaks_func)

    if format_func:
        d["format_func"] = _get(format_func)

    return type(klass_name, (trans,), d)  # type: ignore


@dataclass
class log_trans(trans):
    """
    Create a log transform class for *base*

    Parameters
    ----------
    base : float
        Base for the logarithm. If None, then
        the natural log is used.

    Returns
    -------
    out : type
        Log transform class
    """

    base: float = np.exp(1)
    _: KW_ONLY
    domain: DomainType = (sys.float_info.min, np.inf)

    def __post_init__(self):
        if self.base == 10:
            self._transform = np.log10
        elif self.base == 2:
            self._transform = np.log2
        elif self.base == np.exp(1):
            self._transform = np.log
        else:

            def _transform(x: FloatArrayLike) -> NDArrayFloat:
                return np.log(x) / np.log(self.base)

            self._transform = _transform

        self.breaks_func = breaks_log(base=self.base)
        self.format_func = label_log(base=self.base)
        self.minor_breaks_func = minor_breaks_trans(self, n=int(self.base) - 2)

    def transform(self, x):
        return self._transform(x)

    def inverse(self, x):
        return np.power(self.base, x)


@dataclass
class log10_trans(log_trans):
    """
    Log 10 Transformation
    """

    base: float = 10


@dataclass
class log2_trans(log_trans):
    """
    Log 2 Transformation
    """

    base: float = 2


@dataclass
class exp_trans(trans):
    """
    Create a exponential transform class for *base*

    This is inverse of the log transform.

    Parameters
    ----------
    base : float
        Base of the logarithm

    Returns
    -------
    out : type
        Exponential transform class
    """

    base: float = np.exp(1)

    def transform(self, x):
        return np.power(self.base, x)

    def inverse(self, x):
        return np.log(x) / np.log(self.base)


@dataclass
class log1p_trans(trans):
    """
    Log plus one Transformation
    """

    def transform(self, x):
        return np.log1p(x)

    def inverse(self, x):
        return np.expm1(x)


@dataclass
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

    >>> t = identity_trans(minor_breaks_func=minor_breaks(4))
    >>> t.minor_breaks(major)
    array([0.2, 0.4, 0.6, 0.8, 1.2, 1.4, 1.6, 1.8])
    """

    transform_is_linear: bool = True

    def transform(self, x):
        return x

    def inverse(self, x):
        return x


@dataclass(kw_only=True)
class reverse_trans(trans):
    """
    Reverse Transformation
    """

    transform_is_linear: bool = True

    def transform(self, x):
        return np.negative(x)

    def inverse(self, x):
        return np.negative(x)


@dataclass(kw_only=True)
class sqrt_trans(trans):
    """
    Square-root Transformation
    """

    domain: DomainType = (0, np.inf)

    def transform(self, x):
        return np.sqrt(x)

    def inverse(self, x):
        return np.square(x)


@dataclass(kw_only=True)
class asn_trans(trans):
    """
    Arc-sin square-root Transformation
    """

    transform_is_linear: bool = True

    def transform(self, x: FloatArrayLike) -> NDArrayFloat:
        return 2 * np.arcsin(np.sqrt(x))  # type: ignore

    def inverse(self, x: FloatArrayLike) -> NDArrayFloat:
        x = np.asarray(x)
        return np.sin(x / 2) ** 2  # type: ignore


@dataclass(kw_only=True)
class atanh_trans(trans):
    """
    Arc-tangent Transformation
    """

    transform_is_linear: bool = True

    def transform(self, x):
        return np.arctanh(x)

    def inverse(self, x):
        return np.tanh(x)


@dataclass
class boxcox_trans(trans):
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

    p: float
    offset: int = 0

    def transform(self, x: FloatArrayLike) -> NDArrayFloat:
        x = np.asarray(x)
        if np.any((x + self.offset) < 0):
            raise ValueError(
                "boxcox_trans must be given only positive values. "
                "Consider using modulus_trans instead?"
            )
        if np.abs(self.p) < 1e-7:
            return np.log(x + self.offset)
        else:
            return ((x + self.offset) ** self.p - 1) / self.p

    def inverse(self, x: FloatArrayLike) -> NDArrayFloat:
        x = np.asarray(x)
        if np.abs(self.p) < 1e-7:
            return np.exp(x) - self.offset  # type: ignore
        else:
            return (x * self.p + 1) ** (1 / self.p) - self.offset


@dataclass
class modulus_trans(trans):
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

    p: float
    offset: int = 1

    def transform(self, x: FloatArrayLike) -> NDArrayFloat:
        x = np.asarray(x)
        p, offset = self.p, self.offset

        if np.abs(self.p) < 1e-7:
            return np.sign(x) * np.log(np.abs(x) + offset)  # pyright: ignore[reportReturnType]
        else:
            return np.sign(x) * ((np.abs(x) + offset) ** p - 1) / p

    def inverse(self, x: FloatArrayLike) -> NDArrayFloat:
        x = np.asarray(x)
        p, offset = self.p, self.offset

        if np.abs(self.p) < 1e-7:
            return np.sign(x) * (np.exp(np.abs(x)) - offset)  # type: ignore
        else:
            return np.sign(x) * ((np.abs(x) * p + 1) ** (1 / p) - offset)


@dataclass
class probability_trans(trans):
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

    def __init__(self, distribution: str, *args, **kwargs):
        import scipy.stats as stats

        cdists = {k for k in dir(stats) if hasattr(getattr(stats, k), "cdf")}
        if distribution not in cdists:
            raise ValueError(f"Unknown distribution '{distribution}'")

        self._dist = getattr(stats, distribution)
        self._args = args
        self._kwargs = kwargs

    def transform(self, x: FloatArrayLike) -> NDArrayFloat:
        return self._dist.cdf(x, *self._args, **self._kwargs)

    def inverse(self, x: FloatArrayLike) -> NDArrayFloat:
        return self._dist.ppf(x, *self._args, **self._kwargs)


class logit_trans(probability_trans):
    """
    Logit Transformation
    """

    def __init__(self):
        super().__init__("logistic")


class probit_trans(probability_trans):
    """
    Probit Transformation
    """

    def __init__(self):
        super().__init__("norm")


@dataclass
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

    tz: tzinfo | str | None = None

    _: KW_ONLY
    domain: DomainType = (
        datetime(MINYEAR, 1, 1, tzinfo=UTC),
        datetime(MAXYEAR, 12, 31, tzinfo=UTC),
    )
    breaks_func: BreaksFunction = field(default_factory=breaks_date)
    format_func: FormatFunction = field(default_factory=label_date)

    def __post_init__(self):
        if isinstance(self.tz, str):
            self.tz = ZoneInfo(self.tz)

    def transform(self, x: DatetimeArrayLike) -> NDArrayFloat:  # pyright: ignore[reportIncompatibleMethodOverride]
        """
        Transform from date to a numerical format

        The transform values a unit of [days].
        """
        if not len(x):
            return np.array([])

        try:
            tz = next(iter(x)).tzinfo
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

    def diff_type_to_num(self, x: TimedeltaArrayLike) -> FloatArrayLike:
        """
        Covert timedelta to numerical format

        The timedeltas are converted to a unit of [days].
        """
        return timedelta_to_num(x)


@dataclass(kw_only=True)
class timedelta_trans(trans):
    """
    Timedelta Transformation
    """

    domain: DomainType = (timedelta.min, timedelta.max)
    breaks_func: BreaksFunction = field(default_factory=breaks_timedelta)
    format_func: FormatFunction = field(default_factory=label_timedelta)

    def transform(self, x: TimedeltaArrayLike) -> NDArrayFloat:  # pyright: ignore[reportIncompatibleMethodOverride]
        """
        Transform from Timeddelta to numerical format

        The transform values have a unit of [days]
        """
        return timedelta_to_num(x)

    def inverse(self, x: FloatArrayLike) -> Sequence[pd.Timedelta]:  # pyright: ignore[reportIncompatibleMethodOverride]
        """
        Transform to Timedelta from numerical format
        """
        return num_to_timedelta(x)

    def diff_type_to_num(self, x: TimedeltaArrayLike) -> FloatArrayLike:
        """
        Covert timedelta to numerical format

        The timedeltas are converted to a unit of [days].
        """
        return timedelta_to_num(x)


@dataclass(kw_only=True)
class pd_timedelta_trans(timedelta_trans):
    """
    Pandas timedelta Transformation
    """

    domain: DomainType = (pd.Timedelta.min, pd.Timedelta.max)


class reciprocal_trans(trans):
    """
    Reciprocal Transformation
    """

    def transform(self, x: FloatArrayLike) -> NDArrayFloat:
        return 1 / np.asarray(x)

    def inverse(self, x: FloatArrayLike) -> NDArrayFloat:
        return 1 / np.asarray(x)


@dataclass
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
    """

    sigma: float = 1
    base: float = np.exp(1)

    def transform(self, x: FloatArrayLike) -> NDArrayFloat:
        x = np.asarray(x)
        return np.arcsinh(x / (2 * self.sigma)) / np.log(self.base)

    def inverse(self, x: FloatArrayLike) -> NDArrayFloat:
        x = np.asarray(x)
        return 2 * self.sigma * np.sinh(x * np.log(self.base))

    def minor_breaks(
        self,
        major: FloatArrayLike,
        limits: tuple[float, float] | None = None,
        n: int | None = None,
    ) -> NDArrayFloat:
        n = int(self.base) - 2 if n is None else n
        return super().minor_breaks(major, limits, n)


@dataclass(kw_only=True)
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

    breaks_func: BreaksFunction = breaks_symlog()

    def transform(self, x: FloatArrayLike) -> NDArrayFloat:
        return np.sign(x) * np.log1p(np.abs(x))  # pyright: ignore[reportReturnType]

    def inverse(self, x: FloatArrayLike) -> NDArrayFloat:
        return np.sign(x) * (np.exp(np.abs(x)) - 1)  # type: ignore


def gettrans(t: str | Type[trans] | trans | None = None):
    """
    Return a trans object

    Parameters
    ----------
    t : str | type | trans
        Name of transformation function. If None, returns an
        identity transform.

    Returns
    -------
    out : trans
    """
    if isinstance(t, str):
        names = (f"{t}_trans", t)
        for name in names:
            if t := REGISTRY.get(name):
                return t()
    elif isinstance(t, trans):
        return t
    elif isinstance(t, type) and issubclass(t, trans):
        return t()
    elif t is None:
        return identity_trans()

    raise ValueError(f"Could not get transform object. {t}")
