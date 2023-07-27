from __future__ import annotations

import typing
from dataclasses import dataclass

import numpy as np

from ._colormap import ColorMap
from .hsluv import hex_to_rgb, rgb_to_hex
from .named_colors import get_named_color

if typing.TYPE_CHECKING:
    from typing import Optional, Sequence

    from mizani.typing import (
        NDArrayFloat,
        RGBColor,
        RGBColorArray,
        RGBHexColor,
    )


SPACE256 = np.arange(256)
INNER_SPACE256 = SPACE256[1:-1]
ROUNDING_JITTER = 1e-12

__all__ = ("GradientMap",)


@dataclass
class GradientMap(ColorMap):
    colors: Sequence[RGBHexColor] | Sequence[RGBColor] | RGBColorArray
    values: Optional[Sequence[float]] = None

    def __post_init__(self):
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

    def _generate_colors(
        self, x: NDArrayFloat
    ) -> Sequence[RGBHexColor | None]:
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
            [self._r_lookup[idx], self._g_lookup[idx], self._b_lookup[idx]]
        )
        return [rgb_to_hex(c) for c in arr]


def interp_lookup(x: NDArrayFloat, values: NDArrayFloat) -> NDArrayFloat:
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
    """
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
    lut = np.concatenate(
        [
            [values[0]],
            distance * (values[ind] - values[ind_prev]) + values[ind_prev],
            [values[-1]],
        ]
    )
    return np.clip(lut, 0, 1)
