from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..hsluv import hex_to_rgb, rgb_to_hex
from ._colormap import ColorMap, ColorMapKind

if TYPE_CHECKING:
    from typing import Sequence

    from mizani.typing import (
        FloatArrayLike,
        NDArrayFloat,
        RGBColor,
        RGBColorArray,
        RGBHexColor,
    )


SPACE256 = np.arange(256)
INNER_SPACE256 = SPACE256[1:-1]
ROUNDING_JITTER = 1e-12


class _InterpolatedGen(ColorMap):
    _r_lookup: NDArrayFloat
    _g_lookup: NDArrayFloat
    _b_lookup: NDArrayFloat

    def _generate_colors(self, x: FloatArrayLike) -> Sequence[RGBHexColor]:
        """
        Lookup colors in the interpolated ranges

        Parameters
        ----------
        x :
            Values in the range [0, 1]. O maps to the start of the
            gradient, and 1 to the end of the gradient.
        """
        x = np.asarray(x)
        idx = np.round((x * 255) + ROUNDING_JITTER).astype(int)
        arr = np.column_stack(
            [self._r_lookup[idx], self._g_lookup[idx], self._b_lookup[idx]],
        )
        return [rgb_to_hex(c) for c in arr]


@dataclass
class InterpolatedMap(_InterpolatedGen):
    colors: Sequence[RGBHexColor] | Sequence[RGBColor] | RGBColorArray
    values: Sequence[float] | None = None
    kind: ColorMapKind = ColorMapKind.miscellaneous

    def __post_init__(self):
        from ..named_colors import get_named_color

        if self.values is None:
            values = np.linspace(0, 1, len(self.colors))
        elif len(self.colors) < 2:
            raise ValueError("A color gradient needs two or more colors")
        else:
            values = np.asarray(self.values)
            if values[0] != 0 or values[-1] != 1:
                raise ValueError(
                    "Value points of a color gradient should start"
                    "with 0 and end with 1. "
                    f"Got {values[0]} and {values[-1]}"
                )

        if len(self.colors) != len(values):
            raise ValueError(
                "The values and the colors are different lengths"
                f"colors={len(self.colors)}, values={len(values)}"
            )

        if isinstance(self.colors[0], str):
            colors = [
                hex_to_rgb(get_named_color(c))  # type: ignore
                for c in self.colors
            ]
        else:
            colors = self.colors

        self._data = np.asarray(colors)
        self._r_lookup = interp_lookup(values, self._data[:, 0])
        self._g_lookup = interp_lookup(values, self._data[:, 1])
        self._b_lookup = interp_lookup(values, self._data[:, 2])


def interp_lookup(
    x: NDArrayFloat,
    values: NDArrayFloat,
    values_alt: NDArrayFloat | None = None,
) -> NDArrayFloat:
    """
    Create an interpolation lookup array

    This helps make interpolating between two or more colors
    a discrete task.

    Parameters
    ----------
    x:
        Breaks In the range [0, 1]. Must include 0 and 1 and values
        should be sorted.
    values:
        In the range [0, 1]. Must be the same length as x.
    values_alt:
        In the range [0, 1]. Must be the same length as x.
        Makes it possible to have adjacent interpolation regions
        that with gaps in them numbers. e.g.

            values = [0, 0.1, 0.5, 1]
            values_alt = [0, 0.1, 0.6, 1]

        Creates the regions

            [(0, 0.1), (0.1, 0.5), (0.6, 1)]

        If values_alt is None the region would be

            [(0, 0.1), (0.1, 0.5), (0.5, 1)]
    """
    if values_alt is None:
        values_alt = values

    # - Map x from [0, 1] onto [0, 255] i.e. the color channel
    #   breaks (continuous)
    # - Find where x would be mapped onto the grid (discretizing)
    # - Find the distance between the discrete breaks and the
    #   continuous values of x (with each value scaled by the distance
    #   to previous x value)
    # - Expand the scaled distance (how far to move at each point) to a
    #   value, and move by that scaled distance from the previous point
    x256 = x * 255
    ind = np.searchsorted(x256, SPACE256)[1:-1]
    ind_prev = ind - 1
    distance = (INNER_SPACE256 - x256[ind_prev]) / (x256[ind] - x256[ind_prev])
    stop = values[ind]
    start = values_alt[ind_prev]
    lut = np.concatenate(
        [
            [values[0]],
            start + distance * (stop - start),
            [values[-1]],
        ]
    )
    return np.clip(lut, 0, 1)
