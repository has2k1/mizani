from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ._colormap import ColorMapKind
from ._interpolated import _InterpolatedGen, interp_lookup

if TYPE_CHECKING:
    from mizani.typing import SegmentedColorMapData


@dataclass
class SegmentInterpolatedMap(_InterpolatedGen):
    """
    Gradient colormap by interpolating RGB colors independently

    The input data is the same as Matplotlib's LinearSegmentedColormap
    data.
    """

    data: SegmentedColorMapData
    kind: ColorMapKind = ColorMapKind.miscellaneous

    def __post_init__(self):
        _red = np.asarray(self.data["red"])
        _blue = np.asarray(self.data["blue"])
        _green = np.asarray(self.data["green"])
        self._r_lookup = interp_lookup(*_red.T)
        self._g_lookup = interp_lookup(*_green.T)
        self._b_lookup = interp_lookup(*_blue.T)
        self.colors = self.data
