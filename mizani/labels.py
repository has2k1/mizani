"""
Scales have guides and these are what help users make sense of
the data mapped onto the scale. Common examples of guides include
the x-axis, the y-axis, the keyed legend and a colorbar legend.
The guides have demarcations(breaks), some of which must be labelled.

The `label_*` functions below create functions that convert data
values as understood by a specific scale and return string
representations of those values. Manipulating the string
representation of a value helps improve readability of the guide.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import numpy as np

from .breaks import timedelta_helper
from .utils import (
    match,
    precision,
    round_any,
    same_log10_order_of_magnitude,
)

if TYPE_CHECKING:
    from datetime import datetime, tzinfo
    from typing import Literal, Optional, Sequence

    from mizani.typing import (
        BytesSymbol,
        DurationUnit,
        FloatArrayLike,
        NDArrayTimedelta,
        TupleInt2,
    )

__all__ = [
    "label_comma",
    "label_custom",
    "label_currency",
    "label_dollar",
    "label_percent",
    "label_scientific",
    "label_date",
    "label_number",
    "label_log",
    "label_timedelta",
    "label_pvalue",
    "label_ordinal",
    "label_bytes",
]

UTC = ZoneInfo("UTC")


@dataclass
class label_number:
    """
    Labelling numbers

    Parameters
    ----------
    precision : int
        Number of digits after the decimal point.
    suffix : str
        What to put after the value.
    big_mark : str
        The thousands separator. This is usually
        a comma or a dot.
    decimal_mark : str
        What to use to separate the decimals digits.

    Examples
    --------
    >>> label_number()([.654, .8963, .1])
    ['0.65', '0.90', '0.10']
    >>> label_number(accuracy=0.0001)([.654, .8963, .1])
    ['0.6540', '0.8963', '0.1000']
    >>> label_number(precision=4)([.654, .8963, .1])
    ['0.6540', '0.8963', '0.1000']
    >>> label_number(prefix="$")([5, 24, -42])
    ['$5', '$24', '-$42']
    >>> label_number(suffix="s")([5, 24, -42])
    ['5s', '24s', '-42s']
    >>> label_number(big_mark="_")([1e3, 1e4, 1e5, 1e6])
    ['1_000', '10_000', '100_000', '1_000_000']
    >>> label_number(width=3)([1, 10, 100, 1000])
    ['  1', ' 10', '100', '1000']
    >>> label_number(align="^", width=5)([1, 10, 100, 1000])
    ['  1  ', ' 10  ', ' 100 ', '1000 ']
    >>> label_number(style_positive=" ")([5, 24, -42])
    [' 5', ' 24', '-42']
    >>> label_number(style_positive="+")([5, 24, -42])
    ['+5', '+24', '-42']
    >>> label_number(prefix="$", style_negative="braces")([5, 24, -42])
    ['$5', '$24', '($42)']
    """

    accuracy: Optional[float] = None
    precision: Optional[int] = None
    scale: float = 1
    prefix: str = ""
    suffix: str = ""
    big_mark: str = ""
    decimal_mark: str = "."
    fill: str = ""
    style_negative: Literal["-", "hyphen", "parens"] = "-"
    style_positive: Literal["", "+", " "] = ""
    align: Literal["<", ">", "=", "^"] = ">"
    width: Optional[int] = None

    def __post_init__(self):
        if self.precision is not None:
            if self.accuracy is not None:
                raise ValueError("Specify only one of precision or accuracy")
            self.accuracy = 10**-self.precision

    def __call__(self, x: FloatArrayLike) -> Sequence[str]:
        # Construct formatting according to
        # https://docs.python.org/3/library/string.html#format-string-syntax
        # Specfically using the Format Specification Mini-Language

        # python format only accepts ",", "_" to separate the thousands
        # if we have a non-standard value, we use "," & replace it after
        valid_big_mark = self.big_mark in ("", ",", "_")
        sep = self.big_mark if valid_big_mark else ","

        fmt = (
            f"{self.prefix}" f"{{num:{sep}.{{precision}}f}}" f"{self.suffix}"
        ).format

        x = np.asarray(x)
        x_scaled = x * self.scale

        if self.accuracy is None:
            accuracy = precision(x_scaled)
        else:
            accuracy = self.accuracy

        x = round_any(x, accuracy / self.scale)
        digits = -np.floor(np.log10(accuracy)).astype(int)
        digits = np.minimum(np.maximum(digits, 0), 20)

        res = [fmt(num=abs(n), precision=digits) for n in x_scaled]
        if not valid_big_mark:
            res = [s.replace(",", self.big_mark) for s in res]

        if self.decimal_mark != ".":
            res = [s.replace(".", self.decimal_mark) for s in res]

        pos_fmt = f"{self.style_positive}{{s}}".format

        if self.style_negative == "-":
            neg_fmt = "-{s}".format
        elif self.style_negative == "hyphen":
            neg_fmt = "\u2212{s}".format
        else:
            neg_fmt = "({s})".format

        res = [
            neg_fmt(s=s) if num < 0 else pos_fmt(s=s) for num, s in zip(x, res)
        ]

        if self.width is not None:
            fmt = f"{{s:{self.fill}{self.align}{self.width}}}".format
            res = [fmt(s=s) for s in res]

        return res


@dataclass
class label_custom:
    """
    Creating a custom labelling function

    Parameters
    ----------
    fmt : str, optional
        Format string. Default is the generic new style
        format braces, ``{}``.
    style : 'new' | 'old'
        Whether to use new style or old style formatting.
        New style uses the :meth:`str.format` while old
        style uses ``%``. The format string must be written
        accordingly.

    Examples
    --------
    >>> label = label_custom('{:.2f} USD')
    >>> label([3.987, 2, 42.42])
    ['3.99 USD', '2.00 USD', '42.42 USD']
    """

    fmt: str = "{}"
    style: Literal["old", "new"] = "new"

    def __call__(self, x: FloatArrayLike) -> Sequence[str]:
        """
        Format a sequence of inputs

        Parameters
        ----------
        x : array
            Input

        Returns
        -------
        out : list
            List of strings.
        """
        if self.style == "new":
            return [self.fmt.format(val) for val in x]
        elif self.style == "old":
            return [self.fmt % val for val in x]
        else:
            raise ValueError("style should be either 'new' or 'old'")


# formatting functions
@dataclass
class label_currency(label_number):
    """
    Labelling currencies

    Parameters
    ----------
    prefix : str
        What to put before the value.

    Examples
    --------
    >>> x = [1.232, 99.2334, 4.6, 9, 4500]
    >>> label_currency()(x)
    ['$1.23', '$99.23', '$4.60', '$9.00', '$4500.00']
    >>> label_currency(prefix='C$', precision=0, big_mark=',')(x)
    ['C$1', 'C$99', 'C$5', 'C$9', 'C$4,500']
    """

    prefix: str = "$"

    def __post_init__(self):
        if self.precision is None and self.accuracy is None:
            self.precision = 2
        super().__post_init__()


label_dollar = label_currency
dollar = label_dollar()


@dataclass
class label_comma(label_currency):
    """
    Labels of numbers with commas as separators

    Parameters
    ----------
    precision : int
        Number of digits after the decimal point.

    Examples
    --------
    >>> label_comma()([1000, 2, 33000, 400])
    ['1,000', '2', '33,000', '400']
    """

    prefix: str = ""
    precision: int = 0
    big_mark: str = ","


@dataclass
class label_percent(label_number):
    """
    Labelling percentages

    Multiply by one hundred and display percent sign

    Examples
    --------
    >>> label = label_percent()
    >>> label([.45, 9.515, .01])
    ['45%', '952%', '1%']
    >>> label([.654, .8963, .1])
    ['65%', '90%', '10%']
    """

    scale: float = 100
    suffix: str = "%"


percent = label_percent()


@dataclass
class label_scientific:
    """
    Scientific number labels

    Parameters
    ----------
    digits : int
        Significant digits.

    Examples
    --------
    >>> x = [.12, .23, .34, 45]
    >>> label_scientific()(x)
    ['1.2e-01', '2.3e-01', '3.4e-01', '4.5e+01']

    Notes
    -----
    Be careful when using many digits (15+ on a 64
    bit computer). Consider of the `machine epsilon`_.

    .. _machine epsilon: https://en.wikipedia.org/wiki/Machine_epsilon
    """

    digits: int = 3

    def __post_init__(self):
        tpl = f"{{:.{self.digits}e}}"
        self._label = label_custom(tpl)
        self.trailling_zeros_pattern = re.compile(r"(0+)e")

    def __call__(self, x: FloatArrayLike) -> Sequence[str]:
        if len(x) == 0:
            return []

        def count_zeros(s):
            match = self.trailling_zeros_pattern.search(s)
            if match:
                return len(match.group(1))
            else:
                return 0

        # format and then remove superfluous zeros
        labels = self._label(x)
        n = min([count_zeros(val) for val in labels])
        if n:
            labels = [val.replace("0" * n + "e", "e") for val in labels]
        return labels


scientific = label_scientific()


@dataclass
class label_log:
    """
    Log number labels

    Parameters
    ----------
    base : int
        Base of the logarithm. Default is 10.
    exponent_limits : tuple
        limits (int, int) where if the any of the powers of the
        numbers falls outside, then the labels will be in
        exponent form. This only applies for base 10.
    mathtex : bool
        If True, return the labels in mathtex format as understood
        by Matplotlib.

    Examples
    --------
    >>> label_log()([0.001, 0.1, 100])
    ['0.001', '0.1', '100']

    >>> label_log()([0.0001, 0.1, 10000])
    ['1e-4', '1e-1', '1e4']

    >>> label_log(mathtex=True)([0.0001, 0.1, 10000])
    ['$10^{-4}$', '$10^{-1}$', '$10^{4}$']
    """

    base: float = 10
    exponent_limits: TupleInt2 = (-4, 4)
    mathtex: bool = False

    def _tidyup_labels(self, labels: Sequence[str]) -> Sequence[str]:
        """
        Make all labels uniform in format

        Remove redundant zeros for labels in exponential format.

        Parameters
        ----------
        labels : list-like
            Labels to be tidied.

        Returns
        -------
        out : list-like
            Labels
        """

        def remove_zeroes(s: str) -> str:
            """
            Remove unnecessary zeros for float string s
            """
            tup = s.split("e")
            if len(tup) == 2:
                mantissa = tup[0].rstrip("0").rstrip(".")
                exponent = int(tup[1])
                s = f"{mantissa}e{exponent}" if exponent else mantissa
            return s

        def as_exp(s: str) -> str:
            """
            Float string s as in exponential format
            """
            return s if "e" in s else "{:1.0e}".format(float(s))

        def as_mathtex(s: str) -> str:
            """
            Mathtex for maplotlib
            """
            if "e" not in s:
                assert s == "1", f"Unexpected value {s = }, instead of '1'"
                return f"${self.base}^{{0}}$"

            exp = s.split("e")[1]
            return f"${self.base}^{{{exp}}}$"

        # If any are in exponential format, make all of
        # them expontential
        has_e = ["e" in x for x in labels]
        if not all(has_e) and sum(has_e):
            labels = [as_exp(x) for x in labels]

        labels = [remove_zeroes(x) for x in labels]

        has_e = ["e" in x for x in labels]
        if self.mathtex and any(has_e):
            labels = [as_mathtex(x) for x in labels]

        return labels

    def __call__(self, x: FloatArrayLike) -> Sequence[str]:
        """
        Format a sequence of inputs

        Parameters
        ----------
        x : array
            Input

        Returns
        -------
        out : list
            List of strings.
        """
        if len(x) == 0:
            return []

        # Decide on using exponents
        if self.base == 10:
            xmin = int(np.floor(np.log10(np.min(x))))
            xmax = int(np.ceil(np.log10(np.max(x))))
            emin, emax = self.exponent_limits
            all_multiples = np.all([np.log10(num).is_integer() for num in x])
            beyond_threshold = xmin <= emin or emax <= xmax
            use_exponents = (
                same_log10_order_of_magnitude(x) or all_multiples
            ) and beyond_threshold
            fmt = "{:1.0e}" if use_exponents else "{:g}"
            labels = [fmt.format(num) for num in x]
            return self._tidyup_labels(labels)
        else:

            def _exp(num, base):
                e = np.log(num) / np.log(base)
                e_round = np.round(e)
                e = int(e_round) if np.isclose(e, e_round) else np.round(e, 3)
                return e

            base_txt = f"{self.base}"
            if self.base == np.e:
                base_txt = "e"

            if self.mathtex:
                fmt_parts = (f"${base_txt}^", "{{{e}}}$")
            else:
                fmt_parts = (f"{base_txt}^", "{e}")

            fmt = "".join(fmt_parts)
            exps = [_exp(num, self.base) for num in x]
            labels = [fmt.format(e=e) for e in exps]
            return labels


@dataclass
class label_date:
    """
    Datetime labels

    Parameters
    ----------
    fmt : str
        Format string. See
        :ref:`strftime <strftime-strptime-behavior>`.
    tz : datetime.tzinfo, optional
        Time zone information. If none is specified, the
        time zone will be that of the first date. If the
        first date has no time information then a time zone
        is chosen by other means.

    Examples
    --------
    >>> from datetime import datetime
    >>> x = [datetime(x, 1, 1) for x in [2010, 2014, 2018, 2022]]
    >>> label_date()(x)
    ['2010-01-01', '2014-01-01', '2018-01-01', '2022-01-01']
    >>> label_date('%Y')(x)
    ['2010', '2014', '2018', '2022']

    Can format time

    >>> x = [datetime(2017, 12, 1, 16, 5, 7)]
    >>> label_date("%Y-%m-%d %H:%M:%S")(x)
    ['2017-12-01 16:05:07']

    Time zones are respected

    >>> UTC = ZoneInfo('UTC')
    >>> UG = ZoneInfo('Africa/Kampala')
    >>> x = [datetime(2010, 1, 1, i) for i in [8, 15]]
    >>> x_tz = [datetime(2010, 1, 1, i, tzinfo=UG) for i in [8, 15]]
    >>> label_date('%Y-%m-%d %H:%M')(x)
    ['2010-01-01 08:00', '2010-01-01 15:00']
    >>> label_date('%Y-%m-%d %H:%M')(x_tz)
    ['2010-01-01 08:00', '2010-01-01 15:00']

    Format with a specific time zone

    >>> label_date('%Y-%m-%d %H:%M', tz=UTC)(x_tz)
    ['2010-01-01 05:00', '2010-01-01 12:00']
    >>> label_date('%Y-%m-%d %H:%M', tz='EST')(x_tz)
    ['2010-01-01 00:00', '2010-01-01 07:00']
    """

    fmt: str = "%Y-%m-%d"
    tz: Optional[tzinfo] = None

    def __post_init__(self):
        if isinstance(self.tz, str):
            self.tz = ZoneInfo(self.tz)

    def __call__(self, x: Sequence[datetime]) -> Sequence[str]:
        """
        Format a sequence of inputs

        Parameters
        ----------
        x : array
            Input

        Returns
        -------
        out : list
            List of strings.
        """
        if self.tz is not None:
            x = [d.astimezone(self.tz) for d in x]
        return [d.strftime(self.fmt) for d in x]


@dataclass
class label_timedelta:
    """
    Timedelta labels

    Parameters
    ----------
    units : str, optional
        The units in which the breaks will be computed.
        If None, they are decided automatically. Otherwise,
        the value should be one of::

            'ns'    # nanoseconds
            'us'    # microseconds
            'ms'    # milliseconds
            's'     # seconds
            'min'   # minute
            'h'     # hour
            'day'     # day
            'week'  # week
            'month' # month
            'year'  # year

    show_units : bool
        Whether to append the units symbol to the values.
    zero_has_units : bool
        If True a value of zero
    usetex : bool
        If True, they microseconds identifier string is
        rendered with greek letter *mu*. Default is False.
    space : bool
        If True add a space between the value and the units
    use_plurals : bool
        If True, for the when the value is not 1 and the units are
        one of `week`, `month` and `year`, the plural form of the
        unit is used e.g. `2 weeks`.

    Examples
    --------
    >>> from datetime import timedelta
    >>> x = [timedelta(days=31*i) for i in range(5)]
    >>> label_timedelta()(x)
    ['0 months', '1 month', '2 months', '3 months', '4 months']
    >>> label_timedelta(use_plurals=False)(x)
    ['0 month', '1 month', '2 month', '3 month', '4 month']
    >>> label_timedelta(units='day')(x)
    ['0 days', '31 days', '62 days', '93 days', '124 days']
    >>> label_timedelta(units='day', zero_has_units=False)(x)
    ['0', '31 days', '62 days', '93 days', '124 days']
    >>> label_timedelta(units='day', show_units=False)(x)
    ['0', '31', '62', '93', '124']
    """

    units: Optional[DurationUnit] = None
    show_units: bool = True
    zero_has_units: bool = True
    usetex: bool = False
    space: bool = True
    use_plurals: bool = True

    def __call__(self, x: NDArrayTimedelta) -> Sequence[str]:
        if len(x) == 0:
            return []

        values, units = timedelta_helper.format_info(x, self.units)
        labels = list(label_number()(values))

        if self.show_units:
            if self.usetex and units == "us":
                units = r"$\mu s$"

            if self.use_plurals and units in ("day", "week", "month", "year"):
                units_plural = f"{units}s"
            else:
                units_plural = units

            if self.space:
                units = f" {units}"
                units_plural = f" {units_plural}"
            for i, (num, label) in enumerate(zip(values, labels)):
                if num == 0 and not self.zero_has_units:
                    continue
                elif num == 1:
                    labels[i] = f"{label}{units}"
                else:
                    labels[i] = f"{label}{units_plural}"

        return labels


@dataclass
class label_pvalue:
    """
    p-values labelling

    Parameters
    ----------
    accuracy : float
        Number to round to
    add_p : bool
        Whether to prepend "p=" or "p<" to the output

    Examples
    --------
    >>> x = [.90, .15, .015, .009, 0.0005]
    >>> label_pvalue()(x)
    ['0.9', '0.15', '0.015', '0.009', '<0.001']
    >>> label_pvalue(0.1)(x)
    ['0.9', '0.1', '<0.1', '<0.1', '<0.1']
    >>> label_pvalue(0.1, True)(x)
    ['p=0.9', 'p=0.1', 'p<0.1', 'p<0.1', 'p<0.1']
    """

    accuracy: float = 0.001
    add_p: float = False

    def __call__(self, x: FloatArrayLike) -> Sequence[str]:
        """
        Format a sequence of inputs

        Parameters
        ----------
        x : array
            Input

        Returns
        -------
        out : list
            List of strings.
        """
        x = round_any(x, self.accuracy)
        below = [num < self.accuracy for num in x]

        if self.add_p:
            eq_fmt = "p={:g}".format
            below_label = f"p<{self.accuracy:g}"
        else:
            eq_fmt = "{:g}".format
            below_label = f"<{self.accuracy:g}"

        labels = [below_label if b else eq_fmt(i) for i, b in zip(x, below)]
        return labels


def ordinal(n: float, prefix="", suffix="", big_mark=""):
    # General Case: 0th, 1st, 2nd, 3rd, 4th, 5th, 6th, 7th, 8th, 9th
    # Special Case: 10th, 11th, 12th, 13th
    n = int(n)
    idx = np.min((n % 10, 4))
    _suffix = ("th", "st", "nd", "rd", "th")[idx]
    if 11 <= (n % 100) <= 13:
        _suffix = "th"

    if big_mark:
        s = f"{n:,}"
        if big_mark != ",":
            s = s.replace(",", big_mark)
    else:
        s = f"{n}"

    return f"{prefix}{s}{_suffix}{suffix}"


@dataclass
class label_ordinal:
    """
    Ordinal number labelling

    Parameters
    ----------
    prefix : str
        What to put before the value.
    suffix : str
        What to put after the value.
    big_mark : str
        The thousands separator. This is usually
        a comma or a dot.

    Examples
    --------
    >>> label_ordinal()(range(8))
    ['0th', '1st', '2nd', '3rd', '4th', '5th', '6th', '7th']
    >>> label_ordinal(suffix=' Number')(range(11, 15))
    ['11th Number', '12th Number', '13th Number', '14th Number']
    """

    prefix: str = ""
    suffix: str = ""
    big_mark: str = ""

    def __call__(self, x: FloatArrayLike) -> Sequence[str]:
        labels = [
            ordinal(num, self.prefix, self.suffix, self.big_mark) for num in x
        ]
        return labels


@dataclass
class label_bytes:
    """
    Labelling byte numbers

    Parameters
    ----------
    symbol : str
        Valid symbols are "B", "kB", "MB", "GB", "TB", "PB", "EB",
        "ZB", and "YB" for SI units, and the "iB" variants for
        binary units. Default is "auto" where the symbol to be
        used is determined separately for each value of 1x.
    units : "binary" | "si"
        Which unit base to use, 1024 for "binary" or 1000 for "si".
    fmt : str, optional
        Format sting. Default is ``{:.0f}``.

    Examples
    --------
    >>> x = [1000, 1000000, 4e5]
    >>> label_bytes()(x)
    ['1000 B', '977 KiB', '391 KiB']
    >>> label_bytes(units='si')(x)
    ['1 kB', '1 MB', '400 kB']
    """

    symbol: Literal["auto"] | BytesSymbol = "auto"
    units: Literal["binary", "si"] = "binary"
    fmt: str = "{:.0f} "

    def __post_init__(self):
        if self.units == "si":
            self.base = 1000
            self._all_symbols = (
                "B",
                "kB",
                "MB",
                "GB",
                "TB",
                "PB",
                "EB",
                "ZB",
                "YB",
            )
        else:
            self.base = 1024
            self._all_symbols = (
                "B",
                "KiB",
                "MiB",
                "GiB",
                "TiB",
                "PiB",
                "EiB",
                "ZiB",
                "YiB",
            )

        # possible exponents of base: eg 1000^1, 1000^2, 1000^3, ...
        exponents = np.arange(1, len(self._all_symbols) + 1, dtype=float)
        self._powers = self.base**exponents
        self._validate_symbol(self.symbol, ("auto",) + self._all_symbols)

    def __call__(self, x: FloatArrayLike) -> Sequence[str]:
        _all_symbols = self._all_symbols
        symbol = self.symbol
        if symbol == "auto":
            power = [bisect_right(self._powers, val) for val in x]
            symbols = [_all_symbols[p] for p in power]
        else:
            power = np.array(match([symbol], _all_symbols))
            symbols = [symbol] * len(x)

        x = np.asarray(x)
        power = np.asarray(power, dtype=float)
        values = x / self.base**power
        fmt = (self.fmt + "{}").format
        labels = [fmt(v, s) for v, s in zip(values, symbols)]
        return labels

    def _validate_symbol(self, symbol: str, allowed_symbols: Sequence[str]):
        if symbol not in allowed_symbols:
            raise ValueError(
                "Symbol must be one of {}".format(allowed_symbols)
            )


# Deprecated
comma_format = label_comma
custom_format = label_custom
currency_format = label_currency
label_dollar = label_dollar
percent_format = label_percent
scientific_format = label_scientific
date_format = label_date
number_format = label_number
log_format = label_log
timedelta_format = label_timedelta
pvalue_format = label_pvalue
ordinal_format = label_ordinal
number_bytes_format = label_bytes
