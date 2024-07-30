from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import date, datetime, timedelta, tzinfo
    from types import NoneType
    from typing import (
        Any,
        Callable,
        Literal,
        Optional,
        Protocol,
        Sequence,
        TypeAlias,
        TypedDict,
        TypeVar,
    )

    import numpy as np
    import pandas as pd
    from numpy.typing import NDArray
    from pandas._libs import NaTType

    from mizani._colors._palettes import palette
    from mizani.transforms import trans

    T = TypeVar("T")

    # Tuples
    TupleT2: TypeAlias = tuple[T, T]
    TupleT3: TypeAlias = tuple[T, T, T]
    TupleT4: TypeAlias = tuple[T, T, T, T]
    TupleT5: TypeAlias = tuple[T, T, T, T, T]

    TupleInt2: TypeAlias = TupleT2[int]
    TupleFloat2: TypeAlias = TupleT2[float]
    TupleFloat3: TypeAlias = TupleT3[float]
    TupleFloat4: TypeAlias = TupleT4[float]
    TupleFloat5: TypeAlias = TupleT5[float]
    TupleDatetime2: TypeAlias = TupleT2[datetime]
    TupleDate2: TypeAlias = TupleT2[date]

    # Arrays (strictly numpy)
    NDArrayAny: TypeAlias = NDArray[Any]
    NDArrayBool: TypeAlias = NDArray[np.bool_]
    NDArrayFloat: TypeAlias = NDArray[np.float64]
    NDArrayInt: TypeAlias = NDArray[np.int64]
    NDArrayStr: TypeAlias = NDArray[np.str_]
    NDArrayDatetime: TypeAlias = NDArray[Any]
    NDArrayTimedelta: TypeAlias = NDArray[Any]

    # Series
    AnySeries: TypeAlias = pd.Series[Any]
    BoolSeries: TypeAlias = pd.Series[bool]
    IntSeries: TypeAlias = pd.Series[int]
    FloatSeries: TypeAlias = pd.Series[float]
    DatetimeSeries: TypeAlias = pd.Series[datetime]

    # Use Any as cannot define pd.Series[timedelta]
    TimedeltaSeries: TypeAlias = pd.Series[Any]

    # ArrayLikes
    AnyArrayLike: TypeAlias = NDArrayAny | pd.Series[Any] | Sequence[Any]
    IntArrayLike: TypeAlias = NDArrayInt | IntSeries | Sequence[int]
    FloatArrayLike: TypeAlias = NDArrayFloat | FloatSeries | Sequence[float]
    NumArrayLike: TypeAlias = IntArrayLike | FloatArrayLike
    DatetimeArrayLike: TypeAlias = (
        NDArrayDatetime | DatetimeSeries | Sequence[datetime]
    )
    TimedeltArrayLike: TypeAlias = (
        NDArrayTimedelta | TimedeltaSeries | Sequence[timedelta]
    )

    # Type variable
    TFloatLike = TypeVar("TFloatLike", bound=NDArrayFloat | float)
    TFloatArrayLike = TypeVar("TFloatArrayLike", bound=FloatArrayLike)
    TFloatVector = TypeVar("TFloatVector", bound=NDArrayFloat | FloatSeries)
    TConstrained = TypeVar(
        "TConstrained", int, float, bool, str, complex, datetime, timedelta
    )

    NumericUFunction: TypeAlias = Callable[[TFloatLike], TFloatLike]

    # Nulls for different types
    # float("nan"), np.timedelta64("NaT") & np.datetime64("NaT") do not
    # have distinct types for null
    NullType: TypeAlias = (
        NoneType
        | float
        |
        # Cannot really use NaTType at the moment, e.g. directly
        # instantiating a NaTType is not the same as that from
        # pd.Timestamp("NaT"). Pandas chokes on it.
        NaTType
        | pd.Timestamp
        | pd.Timedelta
        | np.timedelta64
        | np.datetime64
    )

    RGBColor: TypeAlias = tuple[float, float, float] | NDArrayFloat

    RGB256Color: TypeAlias = tuple[int, int, int]
    RGB256Swatch: TypeAlias = Sequence[RGB256Color]
    RGB256Swatches: TypeAlias = Sequence[RGB256Swatch]

    RGBHexColor: TypeAlias = str
    RGBHexSwatch: TypeAlias = Sequence[RGBHexColor]
    RGBHexSwatches: TypeAlias = Sequence[RGBHexSwatch]

    # Change this when numpy gets support for type-hinting shapes
    # Ref: https://github.com/numpy/numpy/issues/16544
    RGBColorArray: TypeAlias = NDArrayFloat

    class SegmentedColorMapData(TypedDict):
        red: Sequence[tuple[float, float, float]]
        green: Sequence[tuple[float, float, float]]
        blue: Sequence[tuple[float, float, float]]

    class SegmentFunctionColorMapData(TypedDict):
        red: Callable[[NDArrayFloat], NDArrayFloat]
        green: Callable[[NDArrayFloat], NDArrayFloat]
        blue: Callable[[NDArrayFloat], NDArrayFloat]

    ColorScheme: TypeAlias = Literal[
        "diverging",
        "qualitative",
        "sequential",
    ]
    ColorSchemeShort: TypeAlias = Literal[
        "div",
        "qual",
        "seq",
    ]
    ColorPalette: TypeAlias = palette

    # Use SI Units where applica
    DurationUnit: TypeAlias = Literal[
        "ns",  # nanosecond
        "us",  # microsecond
        "ms",  # millisecond
        "s",  # second
        "min",  # minute
        "h",  # hour
        "day",  # day
        "week",  # week
        "month",  # month
        "year",  # year
    ]
    Timedelta: TypeAlias = timedelta | pd.Timedelta
    Datetime: TypeAlias = date | datetime | np.datetime64
    SeqDatetime: TypeAlias = (
        Sequence[date] | Sequence[datetime] | Sequence[np.datetime64]
    )
    SeqDatetime64: TypeAlias = Sequence[np.datetime64]
    TzInfo: TypeAlias = tzinfo

    # dateutil.rrule.YEARLY, ..., but not including 2 weekly
    # adding 7 for our own MICROSECONDLY
    DateFreq: TypeAlias = Literal[0, 1, 3, 4, 5, 6, 7]
    DatetimeBreaksUnits: TypeAlias = Literal[
        "auto",
        "microsecond",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
    ]

    ContinuousPalette: TypeAlias = Callable[[FloatArrayLike], Sequence[Any]]
    DiscretePalette: TypeAlias = Callable[[int], NDArrayAny | AnySeries]

    # Mizani
    Trans: TypeAlias = trans
    TransformFunction: TypeAlias = Callable[[AnyArrayLike], NDArrayFloat]
    InverseFunction: TypeAlias = Callable[[FloatArrayLike], NDArrayAny]
    BreaksFunction: TypeAlias = Callable[[tuple[Any, Any]], AnyArrayLike]
    MinorBreaksFunction: TypeAlias = Callable[
        [FloatArrayLike, Optional[TupleFloat2], Optional[int]], NDArrayFloat
    ]

    # Rescale functions
    # This Protocol does not apply to rescale_mid
    class PRescale(Protocol):
        def __call__(
            self,
            x: FloatArrayLike,
            to: TupleFloat2 = (0, 1),
            _from: TupleFloat2 | None = None,
        ) -> NDArrayFloat: ...

    # Censor functions
    class PCensor(Protocol):
        def __call__(
            self,
            x: NDArrayFloat,
            range: TupleFloat2 = (0, 1),
            only_finite: bool = True,
        ) -> NDArrayFloat: ...

    # Any type that has comparison operators can be used to define
    # the domain of a transformation. And implicitly the type of the
    # dataspace.
    class PComparison(Protocol):
        """
        Objects that can be compaired
        """

        def __eq__(self, other, /) -> bool: ...

        def __lt__(self, other, /) -> bool: ...

        def __gt__(self, other, /) -> bool: ...

    DomainType: TypeAlias = TupleT2[PComparison]

    # This does not work probably due to a bug in the typechecker
    # FormatFunction: TypeAlias = Callable[[AnyArrayLike], Sequence[str]]
    FormatFunction: TypeAlias = (
        Callable[[NDArrayAny], Sequence[str]]
        | Callable[[pd.Series[Any]], Sequence[str]]
        | Callable[[Sequence[Any]], Sequence[str]]
    )
    BytesBinarySymbol: TypeAlias = Literal[
        "B",
        "KiB",
        "MiB",
        "GiB",
        "TiB",
        "PiB",
        "EiB",
        "ZiB",
        "YiB",
    ]
    BytesSISymbol: TypeAlias = Literal[
        "B",
        "KB",
        "MB",
        "GB",
        "TB",
        "PB",
        "EB",
        "ZB",
        "YB",
    ]
    BytesSymbol: TypeAlias = BytesBinarySymbol | BytesSISymbol
