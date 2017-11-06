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
from __future__ import division
import sys
from types import FunctionType, MethodType

import numpy as np
import pandas as pd
import datetime
from dateutil import tz
from matplotlib.dates import date2num, num2date

from .external import six
from .breaks import (extended_breaks, log_breaks, minor_breaks,
                     trans_minor_breaks, date_breaks,
                     timedelta_breaks)
from .formatters import mpl_format, date_format, timedelta_format
from .formatters import log_format


__all__ = ['asn_trans', 'atanh_trans', 'boxcox_trans',
           'datetime_trans', 'exp_trans', 'identity_trans',
           'log10_trans', 'log1p_trans', 'log2_trans',
           'log_trans', 'logit_trans', 'probability_trans',
           'probit_trans', 'reverse_trans', 'sqrt_trans',
           'timedelta_trans', 'pd_timedelta_trans',
           'trans', 'trans_new', 'gettrans']


class trans(object):
    """
    Base class for all transforms

    This class is used to transform data and also tell the
    x and y axes how to create and label the tick locations.

    The key methods to override are :meth:`trans.transform`
    and :meth:`trans.inverse`. Alternately, you can quickly
    create a transform class using the :func:`trans_new`
    function.

    Parameters
    ---------
    kwargs : dict
        Attributes of the class to set/override

    Examples
    --------
    By default trans returns one minor break between every pair
    of major break

    >>> major = [0, 1, 2]
    >>> t = trans()
    >>> t.minor_breaks(major)
    array([ 0.5,  1.5])

    Create a trans that returns 4 minor breaks

    >>> t = trans(minor_breaks=minor_breaks(4))
    >>> t.minor_breaks(major)
    array([ 0.2,  0.4,  0.6,  0.8,  1.2,  1.4,  1.6,  1.8])
    """
    #: Aesthetic that the transform works on
    aesthetic = None

    #: Whether the untransformed data is numerical
    dataspace_is_numerical = True

    #: Limits of the transformed data
    domain = (-np.inf, np.inf)

    #: Callable to calculate breaks
    breaks_ = None

    #: Callable to calculate minor_breaks
    minor_breaks = None

    #: Function to format breaks
    format = staticmethod(mpl_format())

    def __init__(self, **kwargs):
        for attr in kwargs:
            if hasattr(self, attr):
                setattr(self, attr, kwargs[attr])
            else:
                raise KeyError(
                    "Unknown Parameter {!r}".format(attr))

        # Defaults
        if (self.breaks_ is None and
                'breaks_' not in kwargs):
            self.breaks_ = extended_breaks(n=5)

        if (self.minor_breaks is None and
                'minor_breaks' not in kwargs):
            self.minor_breaks = minor_breaks(1)

    @staticmethod
    def transform(x):
        """
        Transform of x
        """
        return x

    @staticmethod
    def inverse(x):
        """
        Inverse of x
        """
        return x

    def breaks(self, limits):
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
        vmin = np.max([self.domain[0], limits[0]])
        vmax = np.min([self.domain[1], limits[1]])
        breaks = np.asarray(self.breaks_([vmin, vmax]))

        # Some methods(mpl_breaks, extended_breaks) that
        # calculate breaks take the limits as guide posts and
        # not hard limits.
        breaks = breaks.compress((breaks >= self.domain[0]) &
                                 (breaks <= self.domain[1]))
        return breaks


def trans_new(name, transform, inverse, breaks=None,
              minor_breaks=None, _format=None,
              domain=(-np.inf, np.inf), doc=''):
    """
    Create a transformation class object

    Parameters
    ----------
    name : str
        Name of the transformation
    transform : function
        A function (preferably a `ufunc`) that computes
        the transformation.
    inverse : function
        A function (preferably a `ufunc`) that computes
        the inverse of the transformation.
    breaks : function
        Function to compute the breaks for this transform.
        If None, then a default good enough for a linear
        domain is used.
    minor_breaks : function
        Function to compute the minor breaks for this
        transform. If None, then a default good enough for
        a linear domain is used.
    _format : function
        Function to format the generated breaks.
    domain : array_like
        Domain over which the transformation is valid.
        It should be of length 2.
    doc : str
        Docstring for the class.

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

    klass_name = '{}_trans'.format(name)

    d = {'transform': _get(transform),
         'inverse': _get(inverse),
         'domain': domain,
         '__doc__': doc}

    if breaks:
        d['breaks_'] = _get(breaks)

    if minor_breaks:
        d['minor_breaks'] = _get(minor_breaks)

    if _format:
        d['format'] = _get(_format)

    return type(klass_name, (trans,), d)


def log_trans(base=None, **kwargs):
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
        name = 'log'
        base = np.exp(1)
        transform = np.log
    elif base == 10:
        name = 'log10'
        transform = np.log10
    elif base == 2:
        name = 'log2'
        transform = np.log2
    else:
        name = 'log{}'.format(base)

        def transform(x):
            return np.log(x)/np.log(base)

    # inverse function
    def inverse(x):
        try:
            return base ** x
        except TypeError:
            return [base**val for val in x]

    if 'domain' not in kwargs:
        kwargs['domain'] = (sys.float_info.min, np.inf)

    if 'breaks' not in kwargs:
        kwargs['breaks'] = log_breaks(base=base)

    kwargs['_format'] = log_format(base)

    _trans = trans_new(name, transform, inverse, **kwargs)

    if 'minor_breaks' not in kwargs:
        _trans.minor_breaks = trans_minor_breaks(_trans, 4)

    return _trans


log10_trans = log_trans(10, doc='Log 10 Transformation')
log2_trans = log_trans(2, doc='Log 2 Transformation')


def exp_trans(base=None, **kwargs):
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
        name = 'power_e'
        base = np.exp(1)
    else:
        name = 'power_{}'.format(base)

    # transform function
    def transform(x):
        return base ** x

    # inverse function
    def inverse(x):
        return np.log(x)/np.log(base)

    return trans_new(name, transform, inverse, **kwargs)


class log1p_trans(trans):
    """
    Log plus one Transformation
    """
    transform = staticmethod(np.log1p)
    inverse = staticmethod(np.expm1)


class identity_trans(trans):
    """
    Identity Transformation
    """
    pass


class reverse_trans(trans):
    """
    Reverse Transformation
    """
    transform = staticmethod(np.negative)
    inverse = staticmethod(np.negative)


class sqrt_trans(trans):
    """
    Square-root Transformation
    """
    transform = staticmethod(np.sqrt)
    inverse = staticmethod(np.square)
    domain = (0, np.inf)


class asn_trans(trans):
    """
    Arc-sin square-root Transformation
    """
    @staticmethod
    def transform(x):
        return 2*np.arcsin(np.sqrt(x))

    @staticmethod
    def inverse(x):
        return np.sin(x/2)**2


class atanh_trans(trans):
    """
    Arc-tangent Transformation
    """
    transform = staticmethod(np.arctanh)
    inverse = staticmethod(np.tanh)


def boxcox_trans(p, **kwargs):
    """
    Boxcox Transformation

    Parameters
    ----------
    p : float
        Power parameter, commonly denoted by
        lower-case lambda in formulae
    kwargs : dict
        Keyword arguments passed onto
        :func:`trans_new`. Should not include
        the `transform` or `inverse`.
    """
    if np.abs(p) < 1e-7:
        return log_trans()

    def transform(x):
        return (x**p - 1) / (p * np.sign(x-1))

    def inverse(x):
        return (np.abs(x) * p + np.sign(x)) ** (1 / p)

    kwargs['name'] = kwargs.get('name', 'pow_{}'.format(p))
    kwargs['transform'] = transform
    kwargs['inverse'] = inverse
    return trans_new(**kwargs)


def probability_trans(distribution, *args, **kwargs):
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

    Note
    ----
    Make sure that the distribution is a good enough
    approximation for the data. When this is not the case,
    computations may run into errors. Absence of any errors
    does not imply that the distribution fits the data.
    """
    import scipy.stats as stats
    cdists = {k for k in dir(stats)
              if hasattr(getattr(stats, k), 'cdf')}
    if distribution not in cdists:
        msg = "Unknown distribution '{}'"
        raise ValueError(msg.format(distribution))

    try:
        doc = kwargs.pop('_doc')
    except KeyError:
        doc = ''

    try:
        name = kwargs.pop('_name')
    except KeyError:
        name = 'prob_{}'.format(distribution)

    def transform(x):
        return getattr(stats, distribution).cdf(x, *args, **kwargs)

    def inverse(x):
        return getattr(stats, distribution).ppf(x, *args, **kwargs)

    return trans_new(name,
                     transform, inverse, domain=(0, 1),
                     doc=doc)


logit_trans = probability_trans('logistic', _name='logit',
                                _doc='Logit Transformation')
probit_trans = probability_trans('norm', _name='norm',
                                 _doc='Probit Transformation')


class datetime_trans(trans):
    """
    Datetime Transformation
    """
    dataspace_is_numerical = False
    domain = (datetime.datetime(datetime.MINYEAR, 1, 1,
                                tzinfo=tz.tzutc()),
              datetime.datetime(datetime.MAXYEAR, 12, 31,
                                tzinfo=tz.tzutc()))
    breaks_ = staticmethod(date_breaks())
    format = staticmethod(date_format())

    @staticmethod
    def transform(x):
        """
        Transform from date to a numerical format
        """
        try:
            x = date2num(x)
        except AttributeError:
            # numpy datetime64
            # This is not ideal because the operations do not
            # preserve the np.datetime64 type. May be need
            # a datetime64_trans
            x = [pd.Timestamp(item) for item in x]
            x = date2num(x)
        return x

    @staticmethod
    def inverse(x):
        """
        Transform to date from numerical format
        """
        return num2date(x)


class timedelta_trans(trans):
    """
    Timedelta Transformation
    """
    dataspace_is_numerical = False
    domain = (datetime.timedelta.min, datetime.timedelta.max)
    breaks_ = staticmethod(timedelta_breaks())
    format = staticmethod(timedelta_format())

    @staticmethod
    def transform(x):
        """
        Transform from Timeddelta to numerical format
        """
        # microseconds
        try:
            x = np.array([_x.total_seconds()*10**6 for _x in x])
        except TypeError:
            x = x.total_seconds()*10**6
        return x

    @staticmethod
    def inverse(x):
        """
        Transform to Timedelta from numerical format
        """
        try:
            x = [datetime.timedelta(microseconds=i) for i in x]
        except TypeError:
            x = datetime.timedelta(microseconds=x)
        return x


class pd_timedelta_trans(trans):
    """
    Pandas timedelta Transformation
    """
    dataspace_is_numerical = False
    domain = (pd.Timedelta.min, pd.Timedelta.max)
    breaks_ = staticmethod(timedelta_breaks())
    format = staticmethod(timedelta_format())

    @staticmethod
    def transform(x):
        """
        Transform from Timeddelta to numerical format
        """
        # nanoseconds
        try:
            x = np.array([_x.value for _x in x])
        except TypeError:
            x = x.value
        return x

    @staticmethod
    def inverse(x):
        """
        Transform to Timedelta from numerical format
        """
        try:
            x = [pd.Timedelta(int(i)) for i in x]
        except TypeError:
            x = pd.Timedelta(int(x))
        return x


def gettrans(t):
    """
    Return a trans object

    Parameters
    ----------
    t : string | function | class | trans object
        name of transformation function

    Returns
    -------
    out : trans
    """
    obj = t
    # Make sure trans object is instantiated
    if isinstance(obj, six.string_types):
        name = '{}_trans'.format(obj)
        obj = globals()[name]()
    if isinstance(obj, FunctionType):
        obj = obj()
    if isinstance(obj, type):
        obj = obj()

    if not isinstance(obj, trans):
        raise ValueError("Could not get transform object.")

    return obj
