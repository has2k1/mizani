from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

    from mizani.typing import ColorPalette, ColorScheme, ColorSchemeShort

__all__ = ["get_brewer_palette"]


def get_palette_names(scheme: ColorScheme | ColorSchemeShort) -> list[str]:
    """
    Return list of palette names
    """
    mod = get_palette_module(scheme)
    names = mod.__all__
    return names.copy()


def get_palette_module(scheme: ColorScheme | ColorSchemeShort) -> ModuleType:
    """
    Return Module with the palettes for the scheme
    """
    if scheme in ("sequential", "seq"):
        from . import sequential

        return sequential
    elif scheme in ("qualitative", "qual"):
        from . import qualitative

        return qualitative
    elif scheme in ("diverging", "div"):
        from . import diverging

        return diverging
    else:
        raise ValueError(f"Unknown type of brewer palette: {scheme}")


@cache
def _palette_schemes() -> dict[str, ColorScheme]:
    """
    Return every brewer palette's scheme, keyed by palette name

    Palette names are unique across schemes, so each name identifies one
    scheme.
    """
    schemes: tuple[ColorScheme, ...] = (
        "sequential",
        "qualitative",
        "diverging",
    )
    return {
        name: scheme
        for scheme in schemes
        for name in get_palette_names(scheme)
    }


def get_palette_scheme(name: str) -> ColorScheme:
    """
    Return the scheme for a named palette

    Raise a `ValueError` if the name is not a brewer palette.
    """
    lookup = _palette_schemes()
    if name not in lookup:
        raise ValueError(
            f"Unknown brewer palette: '{name}'. The valid names are "
            f"{sorted(lookup)}."
        )
    return lookup[name]


def number_to_name(scheme: ColorScheme | ColorSchemeShort, n: int) -> str:
    """
    Return palette name that corresponds to a given number

    Uses alphabetical ordering
    """
    mod = get_palette_module(scheme)
    names = mod.__all__
    if n > len(names):
        raise ValueError(
            f"There are only '{len(names)}' palettes of type {scheme}. "
            f"You requested palette no. {n}"
        )
    return names[n - 1]


def get_brewer_palette(
    scheme: ColorScheme | ColorSchemeShort, palette: int | str
) -> ColorPalette:
    """
    Return color palette from a given scheme

    The scheme selects a palette given by number. A palette given by name
    belongs to one scheme only, so the name overrides `scheme`.
    """
    if isinstance(palette, int):
        palette = number_to_name(scheme, palette)
    else:
        scheme = get_palette_scheme(palette)
    mod = get_palette_module(scheme)
    return getattr(mod, palette)
