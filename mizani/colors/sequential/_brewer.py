import numpy as np

from .. import GradientMap
from ..brewer import sequential
from ..color_palette import palette

__all__ = (
    "Blues",
    "BuGn",
    "BuPu",
    "GnBu",
    "Greens",
    "Greys",
    "Oranges",
    "OrRd",
    "PuBu",
    "PuBuGn",
    "PuRd",
    "Purples",
    "RdPu",
    "Reds",
    "YlGn",
    "YlGnBu",
    "YlOrBr",
    "YlOrRd",
)


def _as_gradient_map(name: str) -> GradientMap:
    p: palette = getattr(sequential, name)
    colors = np.asarray(p.swatches[-1], dtype=float) / 255
    return GradientMap(colors)


Blues = _as_gradient_map("Blues")
BuGn = _as_gradient_map("BuGn")
BuPu = _as_gradient_map("BuPu")
GnBu = _as_gradient_map("GnBu")
Greens = _as_gradient_map("Greens")
Greys = _as_gradient_map("Greys")
Oranges = _as_gradient_map("Oranges")
OrRd = _as_gradient_map("OrRd")
PuBu = _as_gradient_map("PuBu")
PuBuGn = _as_gradient_map("PuBuGn")
PuRd = _as_gradient_map("PuRd")
Purples = _as_gradient_map("Purples")
RdPu = _as_gradient_map("RdPu")
Reds = _as_gradient_map("Reds")
YlGn = _as_gradient_map("YlGn")
YlGnBu = _as_gradient_map("YlGnBu")
YlOrBr = _as_gradient_map("YlOrBr")
YlOrRd = _as_gradient_map("YlOrRd")
