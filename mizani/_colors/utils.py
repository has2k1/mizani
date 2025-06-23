from __future__ import annotations

from typing import TYPE_CHECKING, overload

from mizani._colors.named_colors import get_named_color

if TYPE_CHECKING:
    from typing import Any, Sequence, TypeGuard

    from mizani.typing import AnySeries, ColorType, RGBAColor, RGBColor


def float_to_hex(f: float) -> str:
    """
    Convert f in [0, 1] to a hex value in range ["00", "FF"]
    """
    return f"{round(f * 255):02X}"


def is_color_tuple(obj: Any) -> TypeGuard[RGBColor | RGBAColor]:
    """
    Return True if obj a tuple with 3 or 4 floats
    """
    return (
        isinstance(obj, tuple)
        and (len(obj) == 3 or len(obj) == 4)
        and all(isinstance(x, (float, int)) for x in obj)
    )


def rgb_to_hex(t: RGBColor) -> str:
    """
    Convert rgb color tuple to hex
    """
    return "#{:02X}{:02X}{:02X}".format(
        round(t[0] * 255),
        round(t[1] * 255),
        round(t[2] * 255),
    )


def rgba_to_hex(t: RGBAColor) -> str:
    """
    Convert rgba color tuple to hex
    """
    return "#{:02X}{:02X}{:02X}{:02X}".format(
        round(t[0] * 255),
        round(t[1] * 255),
        round(t[2] * 255),
        round(t[3] * 255),
    )


def color_tuple_to_hex(t: RGBColor | RGBAColor) -> str:
    """
    Convert a color tuple to hex
    """
    return rgb_to_hex(t) if len(t) == 3 else rgba_to_hex(t)


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

        if isinstance(alpha, (float, int)):
            c = get_named_color(colors)
            if len(c) > 7:
                return c
            a = float_to_hex(alpha)
            return f"{c}{a}"
        else:
            raise ValueError(f"Expected {alpha=} to be a float.")
    elif is_color_tuple(colors):
        if not isinstance(alpha, (float, int)):
            raise ValueError(f"Expected {alpha=} to be a float.")
        if len(colors) == 3:
            return (*colors, alpha)  # pyright: ignore[reportReturnType]
        else:
            return colors

    if isinstance(alpha, (float, int)):
        return [to_rgba(c, alpha) for c in colors]  # pyright: ignore[reportCallIssue,reportArgumentType]
    else:
        return [to_rgba(c, a) for c, a in zip(colors, alpha)]  # pyright: ignore[reportCallIssue,reportArgumentType]
