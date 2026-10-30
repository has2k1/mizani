"""
Checks for pandas objects that work without pandas installed

mizani never imports pandas. A pandas object can only exist after
something else has imported pandas, so these checks look for pandas in
`sys.modules` and report `False` when it is absent.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import TYPE_CHECKING, Any, overload

import numpy as np

if TYPE_CHECKING:
    from datetime import date, timedelta
    from typing import Sequence, TypeGuard

    import pandas as pd
    from numpy.typing import NDArray

    from mizani.typing import AnySeries, NDArrayAny

__all__ = (
    "get_pandas",
    "is_series",
    "is_categorical_dtype",
    "is_timestamp",
    "is_timedelta",
    "isna",
)


def get_pandas() -> ModuleType | None:
    """
    Return the pandas module if it has been imported, otherwise `None`
    """
    return sys.modules.get("pandas")


def _isinstance(x: Any, name: str) -> bool:
    """
    Return `True` if `x` is an instance of the pandas class `name`
    """
    pd = get_pandas()
    return pd is not None and isinstance(x, getattr(pd, name))


def is_series(x: Any) -> TypeGuard[AnySeries]:
    """
    Return `True` if `x` is a pandas series
    """
    return _isinstance(x, "Series")


def is_categorical_dtype(dtype: Any) -> bool:
    """
    Return `True` if `dtype` is a pandas categorical dtype
    """
    return _isinstance(dtype, "CategoricalDtype")


def is_timestamp(x: Any) -> TypeGuard[pd.Timestamp]:
    """
    Return `True` if `x` is a pandas timestamp
    """
    return _isinstance(x, "Timestamp")


def is_timedelta(x: Any) -> TypeGuard[pd.Timedelta]:
    """
    Return `True` if `x` is a pandas timedelta
    """
    return _isinstance(x, "Timedelta")


def _isna_scalar(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, (float, np.floating)):
        return bool(np.isnan(x))
    if isinstance(x, (np.datetime64, np.timedelta64)):
        return bool(np.isnat(x))
    return False


@overload
def isna(x: AnySeries) -> pd.Series[bool]: ...


@overload
def isna(x: NDArrayAny | Sequence[Any]) -> NDArray[np.bool_]: ...


@overload
def isna(x: float | date | timedelta | np.generic | None) -> bool: ...


def isna(x: Any) -> bool | NDArray[np.bool_] | pd.Series[bool]:
    """
    Detect missing values: `None`, `NaN` and `NaT`

    When pandas is in use, this is `pandas.isna`. A series gives a
    boolean series, any other array-like gives a boolean numpy array,
    and a scalar gives a bool.
    """
    pd = get_pandas()
    if pd is not None:
        return pd.isna(x)

    if np.ndim(x) == 0 and not isinstance(x, np.ndarray):
        return _isna_scalar(x)

    arr = np.asarray(x)
    if arr.dtype.kind in "US" and not isinstance(x, np.ndarray):
        # numpy turns NaN in a list of strings into the string "nan"
        arr = np.asarray(x, dtype=object)
    kind = arr.dtype.kind
    if kind in "fc":
        return np.isnan(arr)
    elif kind in "mM":
        return np.isnat(arr)
    elif kind == "O":
        return np.vectorize(_isna_scalar, otypes=[bool])(arr)
    return np.zeros(arr.shape, dtype=bool)
