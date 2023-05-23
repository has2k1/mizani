from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    import datetime
    from types import NoneType
    from typing import Any, Literal, Sequence, TypeAlias, TypeVar

    import numpy as np
    import numpy.typing as npt
    import pandas as pd
    from pandas._libs import NaTType

    from mizani.colors.color_palette import palette
    from mizani.transforms import trans

    # Tuples
    TupleInt2: TypeAlias = tuple[int, int]
    TupleFloat2: TypeAlias = tuple[float, float]
    TupleFloat3: TypeAlias = tuple[float, float, float]
    TupleFloat4: TypeAlias = tuple[float, float, float, float]
    TupleFloat5: TypeAlias = tuple[float, float, float, float, float]

    # Arrays (strictly numpy)
    NDArrayAny: TypeAlias = npt.NDArray[Any]
    NDArrayBool: TypeAlias = npt.NDArray[np.bool_]
    NDArrayFloat: TypeAlias = npt.NDArray[np.float64]
    NDArrayInt: TypeAlias = npt.NDArray[np.int64]
    NDArrayStr: TypeAlias = npt.NDArray[np.str_]

    # Series
    AnySeries: TypeAlias = pd.Series[Any]
    IntSeries: TypeAlias = pd.Series[int]
    FloatSeries: TypeAlias = pd.Series[float]

    # ArrayLikes
    AnyArrayLike: TypeAlias = NDArrayAny | pd.Series[Any] | Sequence[Any]
    IntArrayLike: TypeAlias = NDArrayInt | IntSeries | Sequence[int]
    FloatArrayLike: TypeAlias = NDArrayFloat | FloatSeries | Sequence[float]
    NumArrayLike: TypeAlias = IntArrayLike | FloatArrayLike

    # Nulls for different types
    # float("nan"), np.timedelta64("NaT") & np.datetime64("NaT") do not
    # have distinct types for null
    NullType = NoneType | NaTType | float | np.timedelta64 | np.datetime64

    # Type Variables
    # A array variable we can pass to a transforming function and expect
    # result to be of the same type
    FloatArrayLikeTV = TypeVar(
        "FloatArrayLikeTV",
        # We cannot use FloatArrayLike type because pyright expect
        # the result to be a FloatArrayLike
        NDArrayFloat,
        FloatSeries,
        Sequence[float],
        TupleFloat2,
    )

    RGB256Color: TypeAlias = tuple[int, int, int]
    RGB256Swatch: TypeAlias = list[RGB256Color]
    RGB256Swatches: TypeAlias = list[RGB256Swatch]

    RGBHexColor: TypeAlias = str
    RGBHexSwatch: TypeAlias = list[RGBHexColor]
    RGBHexSwatches: TypeAlias = list[RGBHexSwatch]

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
    TimedeltaType = pd.Timedelta | datetime.timedelta

    # Mizani
    Trans: TypeAlias = trans
