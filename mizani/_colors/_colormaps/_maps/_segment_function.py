from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mizani._colors import SegmentFunctionMap

if TYPE_CHECKING:
    from typing import Callable

    from mizani.typing import (
        NDArrayFloat,
        SegmentFunctionColorMapData,
    )

__all__ = (
    # miscellaneous
    "afmhot",
    "flag",
    "gist_heat",
    "gnuplot",
    "gnuplot2",
    "ocean",
    "prism",
    "rainbow",
)


# Gnuplot palette functions
def _gpf_32(x: NDArrayFloat) -> NDArrayFloat:
    ret = np.zeros(len(x))
    m = x < 0.25
    ret[m] = 4 * x[m]
    m = (x >= 0.25) & (x < 0.92)
    ret[m] = -2 * x[m] + 1.84
    m = x >= 0.92
    ret[m] = x[m] / 0.08 - 11.5
    return ret


GPF: dict[int, Callable[[NDArrayFloat], NDArrayFloat]] = {
    0: lambda x: np.zeros(len(x)),
    1: lambda x: np.full(len(x), 0.5),
    2: lambda x: np.ones(len(x)),
    3: lambda x: x,
    4: lambda x: x**2,
    5: lambda x: x**3,
    6: lambda x: x**4,
    7: lambda x: np.sqrt(x),
    8: lambda x: np.sqrt(np.sqrt(x)),
    9: lambda x: np.sin(x * np.pi / 2),
    10: lambda x: np.cos(x * np.pi / 2),
    11: lambda x: np.abs(x - 0.5),
    12: lambda x: (2 * x - 1) ** 2,
    13: lambda x: np.sin(x * np.pi),
    14: lambda x: np.abs(np.cos(x * np.pi)),
    15: lambda x: np.sin(x * 2 * np.pi),
    16: lambda x: np.cos(x * 2 * np.pi),
    17: lambda x: np.abs(np.sin(x * 2 * np.pi)),
    18: lambda x: np.abs(np.cos(x * 2 * np.pi)),
    19: lambda x: np.abs(np.sin(x * 4 * np.pi)),
    20: lambda x: np.abs(np.cos(x * 4 * np.pi)),
    21: lambda x: 3 * x,
    22: lambda x: 3 * x - 1,
    23: lambda x: 3 * x - 2,
    24: lambda x: np.abs(3 * x - 1),
    25: lambda x: np.abs(3 * x - 2),
    26: lambda x: (3 * x - 1) / 2,
    27: lambda x: (3 * x - 2) / 2,
    28: lambda x: np.abs((3 * x - 1) / 2),
    29: lambda x: np.abs((3 * x - 2) / 2),
    30: lambda x: x / 0.32 - 0.78125,
    31: lambda x: 2 * x - 0.84,
    32: _gpf_32,
    33: lambda x: np.abs(2 * x - 0.5),
    34: lambda x: 2 * x,
    35: lambda x: 2 * x - 0.5,
    36: lambda x: 2 * x - 1,
}

_afmhot: SegmentFunctionColorMapData = {
    "red": GPF[34],
    "green": GPF[35],
    "blue": GPF[36],
}

_flag: SegmentFunctionColorMapData = {
    "red": lambda x: 0.75 * np.sin((x * 31.5 + 0.25) * np.pi) + 0.5,
    "blue": lambda x: np.sin(x * 31.5 * np.pi),
    "green": lambda x: 0.75 * np.sin((x * 31.5 - 0.25) * np.pi) + 0.5,
}

_gist_heat: SegmentFunctionColorMapData = {
    "red": lambda x: 1.5 * x,
    "green": lambda x: 2 * x - 1,
    "blue": lambda x: 4 * x - 3,
}

_gnuplot: SegmentFunctionColorMapData = {
    "red": GPF[7],
    "green": GPF[5],
    "blue": GPF[15],
}

_gnuplot2: SegmentFunctionColorMapData = {
    "red": GPF[30],
    "green": GPF[31],
    "blue": GPF[32],
}

_ocean: SegmentFunctionColorMapData = {
    "red": GPF[23],
    "green": GPF[28],
    "blue": GPF[3],
}

_prism: SegmentFunctionColorMapData = {
    "red": lambda x: 0.75 * np.sin((x * 20.9 + 0.25) * np.pi) + 0.67,
    "blue": lambda x: 0.75 * np.sin((x * 20.9 - 0.25) * np.pi) + 0.33,
    "green": lambda x: -1.1 * np.sin((x * 20.9) * np.pi),
}

_rainbow: SegmentFunctionColorMapData = {
    "red": GPF[33],
    "green": GPF[13],
    "blue": GPF[10],
}

afmhot = SegmentFunctionMap(_afmhot)
flag = SegmentFunctionMap(_flag)
gist_heat = SegmentFunctionMap(_gist_heat)
gnuplot = SegmentFunctionMap(_gnuplot)
gnuplot2 = SegmentFunctionMap(_gnuplot2)
ocean = SegmentFunctionMap(_ocean)
prism = SegmentFunctionMap(_prism)
rainbow = SegmentFunctionMap(_rainbow)
