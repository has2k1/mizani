"""
According to *On the theory of scales of measurement* by **S.S. Stevens**,
scales can be classified in four ways -- *nominal*, *ordinal*,
*interval* and *ratio*. Using current(2016) terminology, *nominal* data
is made up of unordered categories, *ordinal* data is made up of ordered
categories and the two can be classified as *discrete*. On the other hand
both *interval* and *ratio* data are *continuous*.

The scale classes below show how the rest of the Mizani package can be
used to implement the two categories of scales. The key tasks are
*training* and *mapping* and these correspond to the **train** and
**map** methods.

To train a scale on data means, to make the scale learn the limits of
the data. This is elaborate (or worthy of a dedicated method) for two
reasons:

    - *Practical* -- data may be split up across more than one object,
      yet all will be represented by a single scale.
    - *Conceptual* -- training is a key action that may need to be inserted
      into multiple locations of the data processing pipeline before a
      graphic can be created.

To map data onto a scale means, to associate data values with
values(potential readings) on a scale. This is perhaps the most important
concept unpinning a scale.

The **apply** methods are simple examples of how to put it all together.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from .bounds import censor, rescale
from .utils import (
    CONTINUOUS_KINDS,
    DISCRETE_KINDS,
    get_categories,
    has_dtype,
    match,
    min_max,
)

if TYPE_CHECKING:
    from typing import Any, Sequence, TypeVar

    from mizani.typing import (
        AnyArrayLike,
        Callable,
        ContinuousPalette,
        DiscretePalette,
        FloatArrayLike,
        NDArrayFloat,
        Trans,
    )

    TVector = TypeVar("TVector", NDArrayFloat, pd.Series[float])


__all__ = ["scale_continuous", "scale_discrete"]


class scale_continuous:
    """
    Continuous scale
    """

    @classmethod
    def apply(
        cls,
        x: FloatArrayLike,
        palette: ContinuousPalette,
        na_value: Any = None,
        trans: Trans | None = None,
    ) -> NDArrayFloat:
        """
        Scale data continuously

        Parameters
        ----------
        x : array_like
            Continuous values to scale
        palette : callable ``f(x)``
            Palette to use
        na_value : object
            Value to use for missing values.
        trans : trans
            How to transform the data before scaling. If
            ``None``, no transformation is done.

        Returns
        -------
        out : array_like
            Scaled values
        """
        if trans is not None:
            x = trans.transform(x)

        limits = cls.train(x)
        return cls.map(x, palette, limits, na_value)

    @classmethod
    def train(
        cls, new_data: FloatArrayLike, old: tuple[float, float] | None = None
    ) -> tuple[float, float]:
        """
        Train a continuous scale

        Parameters
        ----------
        new_data : array_like
            New values
        old : array_like
            Old range

        Returns
        -------
        out : tuple
            Limits(range) of the scale
        """
        if old is None:
            old = (-np.inf, np.inf)

        if not len(new_data):
            return old

        new_data = np.asarray(new_data)

        if new_data.dtype.kind not in CONTINUOUS_KINDS:
            raise TypeError("Discrete value supplied to continuous scale")

        new_data = np.hstack([new_data, old])
        return min_max(new_data, na_rm=True, finite=True)

    @classmethod
    def map(
        cls,
        x: FloatArrayLike,
        palette: ContinuousPalette,
        limits: tuple[float, float],
        na_value: Any = None,
        oob: Callable[[TVector], TVector] = censor,
    ) -> NDArrayFloat:
        """
        Map values to a continuous palette

        Parameters
        ----------
        x : array_like
            Continuous values to scale
        palette : callable ``f(x)``
            palette to use
        na_value : object
            Value to use for missing values.
        oob : callable ``f(x)``
            Function to deal with values that are
            beyond the limits

        Returns
        -------
        out : array_like
            Values mapped onto a palette
        """
        x = oob(rescale(x, _from=limits))  # pyright: ignore
        pal = np.asarray(palette(x))
        pal[pd.isna(x)] = na_value
        return pal


class scale_discrete:
    """
    Discrete scale
    """

    @classmethod
    def apply(
        cls,
        x: AnyArrayLike,
        palette: DiscretePalette,
        na_value: Any = None,
    ):
        """
        Scale data discretely

        Parameters
        ----------
        x : array_like
            Discrete values to scale
        palette : callable ``f(x)``
            Palette to use
        na_value : object
            Value to use for missing values.

        Returns
        -------
        out : array_like
            Scaled values
        """
        limits = cls.train(x)
        return cls.map(x, palette, limits, na_value)

    @classmethod
    def train(
        cls,
        new_data: AnyArrayLike,
        old: Sequence[Any] | None = None,
        drop: bool = False,
        na_rm: bool = False,
    ) -> Sequence[Any]:
        """
        Train a continuous scale

        Parameters
        ----------
        new_data : array_like
            New values
        old : array_like
            Old range. List of values known to the scale.
        drop : bool
            Whether to drop(not include) unused categories
        na_rm : bool
            If ``True``, remove missing values. Missing values
            are either ``NaN`` or ``None``.

        Returns
        -------
        out : list
            Values covered by the scale
        """
        old = [] if old is None else list(old)

        if not len(new_data):
            return old

        old_set = set(old)

        # Get the missing values (NaN & Nones) locations and remove them
        nan_bool_idx = pd.isna(new_data)  # type: ignore
        has_na = np.any(nan_bool_idx)

        if not has_dtype(new_data):
            new_data = np.asarray(new_data)

        new_data = cast(np.ndarray, new_data)

        if new_data.dtype.kind not in DISCRETE_KINDS:
            raise TypeError("Continuous value supplied to discrete scale")

        new_data = new_data[~nan_bool_idx]

        # 1. Train i.e. get the new values
        # 2. Update old
        if isinstance(new_data.dtype, pd.CategoricalDtype):
            categories = get_categories(new_data)
            if drop:
                present = set(new_data)
                new = [i for i in categories if i in present]
            else:
                new = list(categories)

            all_set = old_set | set(new)
            ordered_cats = categories.union(old, sort=False)
            limits = [c for c in ordered_cats if c in all_set]
        else:
            new = np.unique(new_data)
            new.sort()

            limits = old + [i for i in new if (i not in old_set)]

        # Add nan if required
        has_na_limits = any(pd.isna(limits))
        if not has_na_limits and not na_rm and has_na:
            limits.append(np.nan)
        return limits

    @classmethod
    def map(
        cls,
        x: AnyArrayLike,
        palette: DiscretePalette,
        limits: Sequence[Any],
        na_value: Any = None,
    ) -> AnyArrayLike:
        """
        Map values to a discrete palette

        Parameters
        ----------
        palette : callable ``f(x)``
            palette to use
        x : array_like
            Continuous values to scale
        na_value : object
            Value to use for missing values.

        Returns
        -------
        out : array_like
            Values mapped onto a palette
        """
        n = len(limits)
        pal = np.asarray(palette(n))[match(x, limits)]
        nas = pd.isna(x)  # type: ignore
        try:
            pal[nas] = na_value
        except TypeError:
            pal = [na_value if isna else v for v, isna in zip(pal, nas)]

        return pal
