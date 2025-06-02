from __future__ import annotations

from typing import TYPE_CHECKING, overload

from mizani._colors.named_colors import get_named_color

if TYPE_CHECKING:
    from typing import Sequence, TypeGuard

    from mizani.typing import AnySeries, ColorType, RGBAColor, RGBColor


def float_to_hex(f: float) -> str:
    """
    Convert f in [0, 1] to a hex value in range ["00", "FF"]
    """
    return f"{round(f * 255):02x}"


def is_rgbcolor(c) -> TypeGuard[RGBColor | RGBAColor]:
    """
    Return True if c is a color tuple
    """
    return isinstance(c, tuple) and isinstance(c[0], (float, int))


@overload
def to_rgba(colors: ColorType, alpha: float) -> ColorType: ...


@overload
def to_rgba(
    colors: Sequence[ColorType], alpha: float
) -> Sequence[ColorType] | ColorType: ...


@overload
def to_rgba(
    colors: Sequence[ColorType], alpha: Sequence[float]
) -> Sequence[ColorType] | ColorType: ...


@overload
def to_rgba(
    colors: AnySeries, alpha: AnySeries | Sequence[float]
) -> Sequence[ColorType] | ColorType: ...


def to_rgba(
    colors: Sequence[ColorType] | AnySeries | ColorType,
    alpha: float | Sequence[float] | AnySeries,
) -> Sequence[ColorType] | ColorType:
    """
    Convert hex colors to rgba values.

    Parameters
    ----------
    colors :
        Color(s) to convert. Note that, if a color is already
        RGBA, it is not modified
    alphas :
        Alpha values

    Returns
    -------
    out :
        RGBA color(s)
    """
    if isinstance(colors, str):
        if colors == "none" or colors == "None":
            return "none"

        if isinstance(alpha, float):
            c = get_named_color(colors)
            if len(c) > 7:
                return c
            a = float_to_hex(alpha)
            return f"{c}{a}"
        else:
            raise ValueError(f"Expected {alpha=} to be a float.")
    elif is_rgbcolor(colors):
        if not isinstance(alpha, (float, int)):
            raise ValueError(f"Expected {alpha=} to be a float.")
        if len(colors) == 3:
            return (*colors, alpha)  # pyright: ignore[reportReturnType]
        elif len(colors) == 4:
            return colors
        else:
            raise ValueError(f"Expected {colors=} to be of length 3.")

    if isinstance(alpha, (float, int)):
        return [to_rgba(c, alpha) for c in colors]  # pyright: ignore[reportCallIssue,reportArgumentType]
    else:
        return [to_rgba(c, a) for c, a in zip(colors, alpha)]  # pyright: ignore[reportCallIssue,reportArgumentType]
