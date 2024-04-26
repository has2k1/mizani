from __future__ import annotations

import typing

from ._colormaps import ColorMap
from ._named_color_values import CRAYON, CSS4, SHORT, XKCD

if typing.TYPE_CHECKING:
    from types import ModuleType

    from mizani.typing import RGBHexColor


__all__ = ("get_colormap", "get_named_color")


class _color_lookup(dict):
    def __getitem__(self, key: str) -> RGBHexColor:
        try:
            return super().__getitem__(key)
        except KeyError as err:
            raise ValueError(f"Unknown name '{key}' for a color.") from err


class _colormap_lookup(dict[str, ColorMap]):
    """
    Lookup (by name) for all available colormaps
    """

    d: dict[str, ColorMap] = {}

    def _lazy_init(self):
        from ._colormaps._maps import (
            _interpolated,
            _listed,
            _palette_interpolated,
            _segment_function,
            _segment_interpolated,
        )

        def _get(mod: ModuleType) -> dict[str, ColorMap]:
            return {name: getattr(mod, name) for name in mod.__all__}

        self.d = {
            **_get(_interpolated),
            **_get(_listed),
            **_get(_palette_interpolated),
            **_get(_segment_function),
            **_get(_segment_interpolated),
        }

    def __getitem__(self, name: str) -> ColorMap:
        if not self.d:
            self._lazy_init()

        try:
            return self.d[name]
        except KeyError as err:
            raise ValueError(f"Unknown colormap: {name}") from err


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
