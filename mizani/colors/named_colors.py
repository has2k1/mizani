from __future__ import annotations

import typing

from ._colormap import ColorMap
from ._named_color_values import CRAYON, CSS4, SHORT, XKCD

if typing.TYPE_CHECKING:
    from mizani.typing import RGBHexColor


__all__ = ("get_colormap", "get_named_color")


class _color_lookup(dict):
    def __getitem__(self, key: str) -> RGBHexColor:
        try:
            return super().__getitem__(key)
        except KeyError as err:
            raise ValueError(f"Unknown name '{key}' for a color.") from err


class _colormap_lookup(dict[str, ColorMap]):
    d: dict[str, ColorMap] = {}

    def _init(self):
        from . import diverging as div
        from . import qualitative as qual
        from . import sequential as seq

        self.d = {
            **{name: getattr(seq, name) for name in seq.__all__},
            **{name: getattr(div, name) for name in div.__all__},
            **{name: getattr(qual, name) for name in qual.__all__},
        }

    def __getitem__(self, name: str) -> ColorMap:
        if not self.d:
            self._init()

        try:
            return self.d[name]
        except KeyError as err:
            raise ValueError(f"Unknow colormap: {name}") from err


NAMED_COLORS = _color_lookup(**SHORT, **CSS4, **XKCD, **CRAYON)


def get_named_color(name: str) -> RGBHexColor:
    """
    Return the Hex code of a color
    """
    if name.startswith("#"):
        return name
    else:
        return NAMED_COLORS[name.lower()]


COLORMAPS = _colormap_lookup()


def get_colormap(name: str) -> ColorMap:
    """
    Return colormap
    """
    return COLORMAPS[name]
