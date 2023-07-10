import numpy as np

from .. import GradientMap
from ..brewer import diverging
from ..color_palette import palette

__all__ = (
    "BrBG",
    "PiYG",
    "PRGn",
    "PuOr",
    "RdBu",
    "RdGy",
    "RdYlBu",
    "RdYlGn",
    "Spectral",
)


def _as_gradient_map(name: str) -> GradientMap:
    p: palette = getattr(diverging, name)
    colors = np.asarray(p.swatches[-1], dtype=float) / 255
    return GradientMap(colors)


BrBG = _as_gradient_map("BrBG")
PiYG = _as_gradient_map("PiYG")
PRGn = _as_gradient_map("PRGn")
PuOr = _as_gradient_map("PuOr")
RdBu = _as_gradient_map("RdBu")
RdGy = _as_gradient_map("RdGy")
RdYlBu = _as_gradient_map("RdYlBu")
RdYlGn = _as_gradient_map("RdYlGn")
Spectral = _as_gradient_map("Spectral")
