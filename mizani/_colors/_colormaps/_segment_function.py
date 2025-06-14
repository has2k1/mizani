from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..hsluv import rgb_to_hex
from ._colormap import ColorMap, ColorMapKind

if TYPE_CHECKING:
    from typing import Sequence

    from mizani.typing import (
        FloatArrayLike,
        RGBHexColor,
        SegmentFunctionColorMapData,
    )


@dataclass
class SegmentFunctionMap(ColorMap):
    """
    Gradient colormap by calculating RGB colors independently

    The input data is the same as Matplotlib's LinearSegmentedColormap
    data whose values for each channel are functions.
    """

    data: SegmentFunctionColorMapData
    kind: ColorMapKind = ColorMapKind.miscellaneous

    def _generate_colors(self, x: FloatArrayLike) -> Sequence[RGBHexColor]:
        x = np.asarray(x)
        arr = np.column_stack(
            [
                self.data["red"](x),
                self.data["blue"](x),
                self.data["green"](x),
            ]
        )
        return [rgb_to_hex(c) for c in arr]  # pyright: ignore[reportArgumentType]
