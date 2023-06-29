from __future__ import annotations

import typing

from ._named_color_values import CRAYON, CSS4, SHORT, XKCD

if typing.TYPE_CHECKING:
    from mizani.typing import RGBHexColor


__all__ = ("get_named_color",)


class _color_lookup(dict):
    def __getitem__(self, key: str) -> RGBHexColor:
        try:
            return super().__getitem__(key)
        except KeyError:
            raise ValueError(f"Unknown name '{key}' for a color.")


NAMED_COLORS = _color_lookup(**SHORT, **CSS4, **XKCD, **CRAYON)


def get_named_color(name: str) -> RGBHexColor:
    """
    Return the Hex code of a color
    """
    if name.startswith("#"):
        return name
    else:
        return NAMED_COLORS[name.lower()]
