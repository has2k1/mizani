from __future__ import annotations

import abc
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Sequence

    from mizani.typing import FloatArrayLike, RGBHexColor

__all__ = ("ColorMap",)


class ColorMapKind(Enum):
    sequential = "sequential"
    diverging = "diverging"
    qualitative = "qualitative"
    cyclic = "cyclic"
    miscellaneous = "miscellaneous"


class ColorMap(abc.ABC):
    """
    Base color for all color maps
    """

    @abc.abstractmethod
    def _generate_colors(self, x: FloatArrayLike) -> Sequence[RGBHexColor]:
        """
        Method to map [0, 1] values onto the a color range

        Subclasses must implement this method

        Parameters
        ----------
        x :
            Values in the range [0, 1]. O maps to the start of the
            gradient, and 1 to the end of the gradient.
        """

    def discrete_palette(self, n: int) -> Sequence[RGBHexColor]:
        """
        Return n colors from the gradient

        Subclasses can override this method

        Parameters
        ----------
        n :
            Number of colors to return from the gradient.
        """
        x = np.linspace(0, 1, n)
        return self._generate_colors(x)

    def continuous_palette(
        self, x: FloatArrayLike
    ) -> Sequence[RGBHexColor | None]:
        """
        Return colors correspondsing to proportions in x

        Parameters
        ----------
        x :
            Values in the range [0, 1]. O maps to the start of the
            gradient, and 1 to the end of the gradient.
        """
        x = np.asarray(x)
        bad_bool_idx = np.isnan(x) | np.isinf(x)
        has_bad = bad_bool_idx.any()

        if has_bad:
            x[bad_bool_idx] = 0

        hex_colors = self._generate_colors(x)

        if has_bad:
            hex_colors = [
                None if isbad else c
                for c, isbad in zip(hex_colors, bad_bool_idx)
            ]
        return hex_colors
