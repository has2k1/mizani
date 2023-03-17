"""
Helper functions for using colorbrewer
"""
from __future__ import annotations

import typing

from palettable import colorbrewer

if typing.TYPE_CHECKING:
    from typing import Iterable

    from mizani.typing import BrewerMapType, BrewerMapTypeAlt


def _first_last(it: Iterable) -> tuple[int, int]:
    """
    First and Last value of iterator as integers
    """
    lst = list(it)
    return int(lst[0]), int(lst[-1])


BREWER_NCOLOR_RANGE = {
    t: {
        palette: _first_last(info.keys())
        for palette, info in colorbrewer.COLOR_MAPS[t].items()
    }
    for t in colorbrewer.COLOR_MAPS
}


def num_colors(map_type: BrewerMapType, palette: str) -> int:
    """
    Number of Colors in Palette
    """
    return BREWER_NCOLOR_RANGE[map_type][palette][1]


def min_num_colors(map_type: BrewerMapType, palette: str) -> int:
    """
    Minimum Number of Colors in Palette
    """
    return BREWER_NCOLOR_RANGE[map_type][palette][0]


def full_map_type_name(
    text: BrewerMapTypeAlt | BrewerMapType,
) -> BrewerMapType:
    """
    Get brewer map_type name from an abbreviation
    """
    lookup: dict[BrewerMapTypeAlt, BrewerMapType] = {
        "div": "Diverging",
        "qual": "Qualitative",
        "seq": "Sequential",
    }
    return lookup.get(text, text).title()  # pyright: ignore


def number_to_name(map_type: BrewerMapType, n: int) -> str:
    """
    Return palette name that corresponds to a given number

    Uses alphabetical ordering
    """
    _n = n - 1
    palettes = sorted(colorbrewer.COLOR_MAPS[map_type].keys())
    if _n < len(palettes):
        return palettes[_n]

    npalettes = len(palettes)
    raise ValueError(
        f"There are only '{npalettes}' palettes of type {map_type}. "
        f"You requested palette no. {n}"
    )


def get_map(name, map_type, number):
    """
    Return a brewer map representation of the specified colormap
    """
    return colorbrewer.get_map(name, map_type, number)
