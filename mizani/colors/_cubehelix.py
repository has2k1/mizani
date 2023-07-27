from __future__ import annotations

import typing
from dataclasses import dataclass

import numpy as np

from ._colormap import ColorMap
from .hsluv import rgb_to_hex

if typing.TYPE_CHECKING:
    from typing import Sequence

    from mizani.typing import (
        FloatArrayLike,
        RGBHexColor,
    )

__all__ = ("CubeHelixMap",)

rotation_matrix = np.array(
    [[-0.14861, +1.78277], [-0.29227, -0.90649], [+1.97294, 0.0]]
)


@dataclass
class CubeHelixMap(ColorMap):
    start: int = 0
    rotation: float = 0.4
    gamma: float = 1.0
    hue: float = 0.8
    light: float = 0.85
    dark: float = 0.15
    reverse: bool = False

    def _generate_colors(self, x: FloatArrayLike) -> Sequence[RGBHexColor]:
        x = np.asarray(x)
        # Apply gamma factor to emphasise low or high intensity values
        xg = x**self.gamma

        # Calculate amplitude and angle of deviation from the black to
        # white diagonal in the plane of constant perceived intensity.
        amplitude = self.hue * xg * (1 - xg) / 2
        phi = 2 * np.pi * (self.start / 3 + self.rotation * x)

        sin_cos = np.array([np.cos(phi), np.sin(phi)])
        rgb = (xg + amplitude * np.dot(rotation_matrix, sin_cos)).T

        if self.reverse:
            rgb = rgb[::-1, :]

        return [rgb_to_hex(c) for c in rgb]

    def discrete_palette(self, n: int) -> Sequence[RGBHexColor]:
        """
        Return n colors from the gradient

        Parameters
        ----------
        n :
            Number of colors to return from the gradient.
        """
        x = np.linspace(self.light, self.dark, n)
        return self._generate_colors(x)
