from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import date, datetime, timedelta
    from types import NoneType
    from typing import (
        Any,
        Callable,
        Literal,
        Protocol,
        Sequence,
        TypeAlias,
        TypedDict,
        TypeVar,
    )

    import numpy as np
    import pandas as pd
    from numpy.typing import NDArray
    from pandas._libs import (
        NaTType,  # pyright: ignore[reportPrivateImportUsage]
    )

    from mizani._colors._palettes import palette
    from mizani.transforms import trans

    T = TypeVar("T")

    # Arrays (strictly numpy)
    NDArrayAny: TypeAlias = NDArray[Any]
    NDArrayFloat: TypeAlias = NDArray[np.floating]
    NDArrayDatetime: TypeAlias = NDArray[np.datetime64]

    # Panda Series
    AnySeries: TypeAlias = pd.Series[Any]
    FloatSeries: TypeAlias = pd.Series[float]
    DatetimeSeries: TypeAlias = pd.Series[datetime]
    TimedeltaSeries: TypeAlias = pd.Series[pd.Timedelta]

    # ArrayLikes
    AnyArrayLike: TypeAlias = NDArrayAny | pd.Series[Any] | Sequence[Any]
    FloatArrayLike: TypeAlias = NDArrayFloat | FloatSeries | Sequence[float]
    DatetimeArrayLike: TypeAlias = (
        NDArrayDatetime | DatetimeSeries | Sequence[datetime]
    )
    TimedeltaArrayLike: TypeAlias = (
        Sequence[timedelta] | Sequence[pd.Timedelta] | TimedeltaSeries
    )

    # Type variable
    TFloatLike = TypeVar("TFloatLike", NDArrayFloat, float)
    TFloatArrayLike = TypeVar("TFloatArrayLike", bound=FloatArrayLike)
    TFloatVector = TypeVar("TFloatVector", NDArrayFloat, FloatSeries)

    NumericUFunction: TypeAlias = Callable[[TFloatLike], TFloatLike]

    # Nulls for different types
    # float("nan"), np.timedelta64("NaT") & np.datetime64("NaT") do not
    # have distinct types for null
    NullType: TypeAlias = (
        NoneType
        | float
        # Cannot really use NaTType at the moment, e.g. directly
        # instantiating a NaTType is not the same as that from
        # pd.Timestamp("NaT"). Pandas chokes on it.
        | NaTType
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
    SeqTimedelta: TypeAlias = Sequence[timedelta] | Sequence[pd.Timedelta]

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
        [FloatArrayLike, tuple[float, float] | None, int | None],
        NDArrayFloat,
    ]

    # Rescale functions
    # This Protocol does not apply to rescale_mid
    class PRescale(Protocol):
        def __call__(
            self,
            x: FloatArrayLike,
            to: tuple[float, float] = (0, 1),
            _from: tuple[float, float] | None = None,
        ) -> NDArrayFloat: ...

    # Censor functions
    class PCensor(Protocol):
        def __call__(
            self,
            x: NDArrayFloat,
            range: tuple[float, float] = (0, 1),
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

    DomainType: TypeAlias = tuple[PComparison, PComparison]

    TFloatTimedelta = TypeVar("TFloatTimedelta", float, timedelta)

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
