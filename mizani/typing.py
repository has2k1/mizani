from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from datetime import date, datetime, timedelta, tzinfo
    from types import NoneType
    from typing import (
        Any,
        Callable,
        Literal,
        Optional,
        Sequence,
        TypeAlias,
        TypeVar,
    )

    import numpy as np
    import pandas as pd
    from numpy.typing import NDArray
    from pandas._libs import NaTType

    from mizani.colors.color_palette import palette
    from mizani.transforms import trans

    T = TypeVar("T")

    Int: TypeAlias = int | np.int64
    Float: TypeAlias = float | np.float64

    # Tuples
    TupleT2: TypeAlias = tuple[T, T]
    TupleT3: TypeAlias = tuple[T, T, T]
    TupleT4: TypeAlias = tuple[T, T, T, T]
    TupleT5: TypeAlias = tuple[T, T, T, T, T]

    TupleInt2: TypeAlias = TupleT2[int]
    TupleFloat2: TypeAlias = TupleT2[float] | TupleT2[np.float64]
    TupleFloat3: TypeAlias = TupleT3[float] | TupleT3[np.float64]
    TupleFloat4: TypeAlias = TupleT4[float] | TupleT4[np.float64]
    TupleFloat5: TypeAlias = TupleT5[float] | TupleT5[np.float64]

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
    IntSeries: TypeAlias = pd.Series[int]
    FloatSeries: TypeAlias = pd.Series[float]

    # Sequences that support vectorized operations
    IntVector: TypeAlias = NDArrayInt | IntSeries
    FloatVector: TypeAlias = NDArrayFloat | FloatSeries
    AnyVector: TypeAlias = NDArrayAny | AnySeries
    NumVector: TypeAlias = IntVector | FloatVector

    # ArrayLikes
    AnyArrayLike: TypeAlias = NDArrayAny | pd.Series[Any] | Sequence[Any]
    IntArrayLike: TypeAlias = NDArrayInt | IntSeries | Sequence[int]
    FloatArrayLike: TypeAlias = NDArrayFloat | FloatSeries | Sequence[float]
    NumArrayLike: TypeAlias = IntArrayLike | FloatArrayLike

    NumericUFunction: TypeAlias = (
        Callable[[FloatVector], FloatVector]
        | Callable[[IntVector], FloatVector]
        | Callable[[float], float]
        | Callable[[int], float]
    )

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
    RGB256Swatch: TypeAlias = list[RGB256Color]
    RGB256Swatches: TypeAlias = list[RGB256Swatch]

    RGBHexColor: TypeAlias = str
    RGBHexSwatch: TypeAlias = list[RGBHexColor]
    RGBHexSwatches: TypeAlias = list[RGBHexSwatch]

    # Change this when numpy gets support for type-hinting shapes
    # Ref: https://github.com/numpy/numpy/issues/16544
    RGBColorArray: TypeAlias = NDArrayFloat

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

    DurationUnit: TypeAlias = Literal[
        "ns",  # nanosecond
        "us",  # microsecond
        "ms",  # millisecond
        "s",  # second
        "m",  # month
        "h",  # hour
        "d",  # day
        "w",  # week
        "M",  # month
        "y",  # year
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
    TransformFunction: TypeAlias = Callable[[AnyVector], FloatVector]
    InverseFunction: TypeAlias = Callable[[FloatVector], AnyVector]
    BreaksFunction: TypeAlias = Callable[[tuple[Any, Any]], AnyArrayLike]
    MinorBreaksFunction: TypeAlias = Callable[
        [FloatVector, Optional[TupleFloat2], Optional[int]], FloatVector
    ]

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
