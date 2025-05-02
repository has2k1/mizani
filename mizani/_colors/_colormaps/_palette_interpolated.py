from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._interpolated import InterpolatedMap

if TYPE_CHECKING:
    from mizani._colors._palettes import palette


class PaletteInterpolatedMap(InterpolatedMap):
    """
    Make a colormap by interpolating the colors of palette
    """

    def __init__(self, palette: palette):
        colors = np.asarray(palette.swatches[-1], dtype=float) / 255
        super().__init__(colors, kind=palette.kind)
        self.palette = palette

    def discrete_palette(self, n):
        """
        Pick exact colors from the swatch if possible
        """
        if self.palette.min_colors <= n <= self.palette.max_colors:
            return self.palette.get_hex_swatch(n)
        return super().discrete_palette(n)
