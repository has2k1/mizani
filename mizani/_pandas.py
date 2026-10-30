"""
Optional pandas support

pandas is not a requirement of mizani. When the values that mizani
works with come from pandas (e.g. a pandas series), pandas has been
imported and mizani uses it. mizani never imports pandas itself.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from typing import TypeGuard

    import pandas as pd

    from mizani.typing import AnySeries

__all__ = (
    "get_pandas",
    "is_series",
    "is_categorical_dtype",
    "is_timedelta",
    "isna",
)


def get_pandas() -> ModuleType | None:
    """
    Return pandas if it has been imported, otherwise None
    """
    return sys.modules.get("pandas")


def is_series(x: Any) -> TypeGuard[AnySeries]:
    """
    Return True if x is a pandas series
    """
    pd = get_pandas()
    return pd is not None and isinstance(x, pd.Series)


def is_categorical_dtype(dtype: Any) -> bool:
    """
    Return True if dtype is a pandas categorical dtype
    """
    pd = get_pandas()
    return pd is not None and isinstance(dtype, pd.CategoricalDtype)


def is_timedelta(x: Any) -> TypeGuard[pd.Timedelta]:
    """
    Return True if x is a pandas timedelta
    """
    pd = get_pandas()
    return pd is not None and isinstance(x, pd.Timedelta)


def _isna_scalar(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, (float, np.floating)):
        return bool(np.isnan(x))
    if isinstance(x, (np.datetime64, np.timedelta64)):
        return bool(np.isnat(x))
    return False


def isna(x: Any) -> Any:
    """
    Detect missing values (None, NaN and NaT)

    This is equivalent to `pandas.isna`, which is used when pandas
    is in use. For a scalar it returns a bool, otherwise a boolean
    numpy array.
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
