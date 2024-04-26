"""
Palettes are the link between data values and the values along
the dimension of a scale. Before a collection of values can be
represented on a scale, they are transformed by a palette. This
transformation is knowing as mapping. Values are mapped onto a
scale by a palette.

Scales tend to have restrictions on the magnitude of quantities
that they can intelligibly represent. For example, the size of
a point should be significantly smaller than the plot panel
onto which it is plotted or else it would be hard to compare
two or more points. Therefore palettes must be created that
enforce such restrictions. This is the reason for the ``*_pal``
functions that create and return the actual palette functions.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol
from warnings import warn

import numpy as np

from ._colors import (
    CubeHelixMap,
    InterpolatedMap,
    get_colormap,
    get_named_color,
    hex_to_rgb,
    hsluv,
    rgb_to_hex,
)
from .bounds import rescale
from .utils import identity

if TYPE_CHECKING:
    from typing import Any, Literal, Optional, Sequence

    from mizani.typing import (
        Callable,
        ColorScheme,
        ColorSchemeShort,
        FloatArrayLike,
        NDArrayFloat,
        RGBHexColor,
        TupleFloat2,
        TupleFloat3,
    )


__all__ = [
    "hls_palette",
    "husl_palette",
    "rescale_pal",
    "area_pal",
    "abs_area",
    "grey_pal",
    "hue_pal",
    "brewer_pal",
    "gradient_n_pal",
    "cmap_pal",
    "cmap_d_pal",
    "desaturate_pal",
    "manual_pal",
    "xkcd_palette",
    "crayon_palette",
    "cubehelix_pal",
]


class _discrete_pal(Protocol):
    """
    Discrete palette maker
    """

    def __call__(self, n: int) -> Sequence[Any]:
        """
        Palette method
        """
        ...


class _continuous_pal(Protocol):
    """
    Continuous palette maker
    """

    def __call__(self, x: Sequence[Any]) -> NDArrayFloat:
        """
        Palette method
        """
        ...


class _continuous_color_pal(Protocol):
    """
    Continuous color palette maker
    """

    def __call__(self, x: FloatArrayLike) -> Sequence[RGBHexColor | None]:
        """
        Palette method
        """
        ...


def hls_palette(
    n_colors: int = 6, h: float = 0.01, l: float = 0.6, s: float = 0.65
) -> Sequence[TupleFloat3]:
    """
    Get a set of evenly spaced colors in HLS hue space.

    h, l, and s should be between 0 and 1

    Parameters
    ----------

    n_colors : int
        number of colors in the palette
    h : float
        first hue
    l : float
        lightness
    s : float
        saturation

    Returns
    -------
    palette : list
        List of colors as RGB hex strings.

    See Also
    --------
    husl_palette : Make a palette using evenly spaced circular
        hues in the HUSL system.

    Examples
    --------
    >>> len(hls_palette(2))
    2
    >>> len(hls_palette(9))
    9
    """
    hues = np.linspace(0, 1, n_colors + 1)[:-1]
    hues += h
    hues %= 1
    hues -= hues.astype(int)
    palette = [colorsys.hls_to_rgb(h_i, l, s) for h_i in hues]
    return palette


def husl_palette(
    n_colors: int = 6, h: float = 0.01, s: float = 0.9, l: float = 0.65
) -> Sequence[TupleFloat3]:
    """
    Get a set of evenly spaced colors in HUSL hue space.

    h, s, and l should be between 0 and 1

    Parameters
    ----------

    n_colors : int
        number of colors in the palette
    h : float
        first hue
    s : float
        saturation
    l : float
        lightness

    Returns
    -------
    palette : list
        List of colors as RGB hex strings.

    See Also
    --------
    hls_palette : Make a palette using evenly spaced circular
        hues in the HSL system.

    Examples
    --------
    >>> len(husl_palette(3))
    3
    >>> len(husl_palette(11))
    11
    """
    hues = np.linspace(0, 1, n_colors + 1)[:-1]
    hues += h
    hues %= 1
    hues *= 359
    s *= 99
    l *= 99
    palette = [hsluv.hsluv_to_rgb((h_i, s, l)) for h_i in hues]
    return palette


@dataclass
class rescale_pal(_continuous_pal):
    """
    Rescale the input to the specific output range.

    Useful for alpha, size, and continuous position.

    Parameters
    ----------
    range : tuple
        Range of the scale

    Returns
    -------
    out : function
        Palette function that takes a sequence of values
        in the range ``[0, 1]`` and returns values in
        the specified range.

    Examples
    --------
    >>> palette = rescale_pal()
    >>> palette([0, .2, .4, .6, .8, 1])
    array([0.1 , 0.28, 0.46, 0.64, 0.82, 1.  ])

    The returned palette expects inputs in the ``[0, 1]``
    range. Any value outside those limits is clipped to
    ``range[0]`` or ``range[1]``.

    >>> palette([-2, -1, 0.2, .4, .8, 2, 3])
    array([0.1 , 0.1 , 0.28, 0.46, 0.82, 1.  , 1.  ])
    """

    range: TupleFloat2 = (0.1, 1)

    def __call__(self, x: FloatArrayLike) -> NDArrayFloat:
        return rescale(x, self.range, _from=(0, 1))


@dataclass
class area_pal(_continuous_pal):
    """
    Point area palette (continuous).

    Parameters
    ----------
    range : tuple
        Numeric vector of length two, giving range of possible sizes.
        Should be greater than 0.

    Returns
    -------
    out : function
        Palette function that takes a sequence of values
        in the range ``[0, 1]`` and returns values in
        the specified range.

    Examples
    --------
    >>> x = np.arange(0, .6, .1)**2
    >>> palette = area_pal()
    >>> palette(x)
    array([1. , 1.5, 2. , 2.5, 3. , 3.5])

    The results are equidistant because the input ``x`` is in
    area space, i.e it is squared.
    """

    range: TupleFloat2 = (1, 6)

    def __call__(self, x: FloatArrayLike) -> NDArrayFloat:
        return rescale(np.sqrt(x), to=self.range, _from=(0, 1))


@dataclass
class abs_area(_continuous_pal):
    """
    Point area palette (continuous), with area proportional to value.

    Parameters
    ----------
    max : float
        A number representing the maximum size

    Returns
    -------
    out : function
        Palette function that takes a sequence of values
        in the range ``[0, 1]`` and returns values in the range
        ``[0, max]``.

    Examples
    --------
    >>> x = np.arange(0, .8, .1)**2
    >>> palette = abs_area(5)
    >>> palette(x)
    array([0. , 0.5, 1. , 1.5, 2. , 2.5, 3. , 3.5])

    Compared to :func:`area_pal`, :func:`abs_area` will handle values
    in the range ``[-1, 0]`` without returning ``np.nan``. And values
    whose absolute value is greater than 1 will be clipped to the
    maximum.
    """

    max: float

    def __call__(self, x: FloatArrayLike) -> NDArrayFloat:
        return rescale(np.sqrt(np.abs(x)), to=(0, self.max), _from=(0, 1))


@dataclass
class grey_pal(_discrete_pal):
    """
    Utility for creating continuous grey scale palette

    Parameters
    ----------
    start : float
        grey value at low end of palette
    end : float
        grey value at high end of palette

    Returns
    -------
    out : function
        Continuous color palette that takes a single
        :class:`int` parameter ``n`` and returns ``n``
        equally spaced colors.

    Examples
    --------
    >>> palette = grey_pal()
    >>> palette(5)
    ['#333333', '#737373', '#989898', '#b4b4b4', '#cccccc']
    """

    start: float = 0.2
    end: float = 0.8

    def __post_init__(self):
        start, end = self.start, self.end
        colors = (start, start, start), (end, end, end)
        self._cmap = InterpolatedMap(colors)

    def __call__(self, n: int) -> Sequence[RGBHexColor | None]:
        gamma = 2.2
        # The grey scale points are linearly separated in
        # gamma encoded space
        space = np.linspace(self.start**gamma, self.end**gamma, n)
        # Map points onto the [0, 1] palette domain
        x = (space ** (1.0 / gamma) - self.start) / (self.end - self.start)
        return self._cmap.continuous_palette(x)


@dataclass
class hue_pal(_discrete_pal):
    """
    Utility for making hue palettes for color schemes.

    Parameters
    ----------
    h : float
        first hue. In the [0, 1] range
    l : float
        lightness. In the [0, 1] range
    s : float
        saturation. In the [0, 1] range
    color_space : 'hls' | 'husl'
        Color space to use for the palette

    Returns
    -------
    out : function
        A discrete color palette that takes a single
        :class:`int` parameter ``n`` and returns ``n``
        equally spaced colors. Though the palette
        is continuous, since it is varies the hue it
        is good for categorical data. However if ``n``
        is large enough the colors show continuity.

    Examples
    --------
    >>> hue_pal()(5)
    ['#db5f57', '#b9db57', '#57db94', '#5784db', '#c957db']
    >>> hue_pal(color_space='husl')(5)
    ['#e0697e', '#9b9054', '#569d79', '#5b98ab', '#b675d7']
    """

    h: float = 0.01
    l: float = 0.6
    s: float = 0.65
    color_space: Literal["hls", "husl"] = "hls"

    def __post_init__(self):
        h, l, s = self.h, self.l, self.s
        if not all(0 <= val <= 1 for val in (h, l, s)):
            msg = (
                "hue_pal expects values to be between 0 and 1. "
                f"I got {h=}, {l=}, {s=}"
            )
            raise ValueError(msg)

        if self.color_space not in ("hls", "husl"):
            msg = "color_space should be one of ['hls', 'husl']"
            raise ValueError(msg)

    def __call__(self, n: int) -> Sequence[RGBHexColor]:
        lookup = {"husl": husl_palette, "hls": hls_palette}
        palette = lookup[self.color_space]
        colors = palette(n, h=self.h, l=self.l, s=self.s)
        return [hsluv.rgb_to_hex(c) for c in colors]


@dataclass
class brewer_pal(_discrete_pal):
    """
    Utility for making a brewer palette

    Parameters
    ----------
    type : 'sequential' | 'qualitative' | 'diverging'
        Type of palette. Sequential, Qualitative or
        Diverging. The following abbreviations may
        be used, ``seq``, ``qual`` or ``div``.
    palette : int | str
        Which palette to choose from. If is an integer,
        it must be in the range ``[0, m]``, where ``m``
        depends on the number sequential, qualitative or
        diverging palettes. If it is a string, then it
        is the name of the palette.
    direction : int
        The order of colours in the scale. If -1 the order
        of colors is reversed. The default is 1.

    Returns
    -------
    out : function
        A color palette that takes a single
        :class:`int` parameter ``n`` and returns ``n``
        colors. The maximum value of ``n`` varies
        depending on the parameters.

    Examples
    --------
    >>> brewer_pal()(5)
    ['#EFF3FF', '#BDD7E7', '#6BAED6', '#3182BD', '#08519C']
    >>> brewer_pal('qual')(5)
    ['#7FC97F', '#BEAED4', '#FDC086', '#FFFF99', '#386CB0']
    >>> brewer_pal('qual', 2)(5)
    ['#1B9E77', '#D95F02', '#7570B3', '#E7298A', '#66A61E']
    >>> brewer_pal('seq', 'PuBuGn')(5)
    ['#F6EFF7', '#BDC9E1', '#67A9CF', '#1C9099', '#016C59']

    The available color names for each palette type can be
    obtained using the following code::

        from mizani._colors.brewer import get_palette_names

        print(get_palette_names("sequential"))
        print(get_palette_names("qualitative"))
        print(get_palette_names("diverging"))
    """

    type: ColorScheme | ColorSchemeShort = "seq"
    palette: int | str = 1
    direction: Literal[1, -1] = 1

    def __post_init__(self):
        from mizani._colors._palettes.brewer import get_brewer_palette

        if self.direction not in (1, -1):
            raise ValueError("direction should be 1 or -1.")

        self.bpal = get_brewer_palette(self.type, self.palette)

    def __call__(self, n: int) -> Sequence[RGBHexColor | None]:
        # Only draw the maximum allowable colors from the palette
        # and fill any remaining spots with None
        _n = min(max(n, self.bpal.min_colors), self.bpal.max_colors)
        color_map = self.bpal.get_hex_swatch(_n)
        colors = color_map[:n]
        if n > self.bpal.max_colors:
            msg = (
                "Warning message:"
                f"Brewer palette {self.bpal.name} has a maximum "
                f"of {self.bpal.max_colors} colors Returning the "
                "palette you asked for with that many colors"
            )
            warn(msg)
            colors = list(colors) + [None] * (n - self.bpal.max_colors)
        return colors[:: self.direction]


@dataclass
class gradient_n_pal(_continuous_color_pal):
    """
    Create a n color gradient palette

    Parameters
    ----------
    colors : list
        list of colors
    values : list, optional
        list of points in the range [0, 1] at which to
        place each color. Must be the same size as
        `colors`. Default to evenly space the colors

    Returns
    -------
    out : function
        Continuous color palette that takes a single
        parameter either a :class:`float` or a sequence
        of floats maps those value(s) onto the palette
        and returns color(s). The float(s) must be
        in the range [0, 1].

    Examples
    --------
    >>> palette = gradient_n_pal(['red', 'blue'])
    >>> palette([0, .25, .5, .75, 1])
    ['#ff0000', '#bf0040', '#7f0080', '#4000bf', '#0000ff']
    >>> palette([-np.inf, 0, np.nan, 1, np.inf])
    [None, '#ff0000', None, '#0000ff', None]
    """

    colors: Sequence[str]
    values: Optional[Sequence[float]] = None

    def __post_init__(self):
        self._cmap = InterpolatedMap(self.colors, self.values)

    def __call__(self, x: FloatArrayLike) -> Sequence[RGBHexColor | None]:
        return self._cmap.continuous_palette(x)


@dataclass
class cmap_pal(_continuous_color_pal):
    """
    Create a continuous palette using a colormap

    Parameters
    ----------
    name : str
        Name of colormap

    Returns
    -------
    out : function
        Continuous color palette that takes a single
        parameter either a :class:`float` or a sequence
        of floats maps those value(s) onto the palette
        and returns color(s). The float(s) must be
        in the range [0, 1].

    Examples
    --------
    >>> palette = cmap_pal('viridis')
    >>> palette([.1, .2, .3, .4, .5])
    ['#482475', '#414487', '#355f8d', '#2a788e', '#21918c']
    """

    name: str

    def __post_init__(self):
        self._direction: Literal[1, -1] = 1
        # Accomodate matplotlib naming convention and allow
        # names that end with _r to return reversed colormaps
        if self.name.endswith("_r"):
            self._direction = -1
            self.name = self.name[:-2]
        self.cm = get_colormap(self.name)

    def __call__(self, x: FloatArrayLike) -> Sequence[RGBHexColor | None]:
        if self._direction == -1:
            x = 1.0 - np.asarray(x)
        return self.cm.continuous_palette(x)


@dataclass
class cmap_d_pal(_discrete_pal):
    """
    Create a discrete palette from a colormap

    Parameters
    ----------
    name : str
        Name of colormap

    Returns
    -------
    out : function
        A discrete color palette that takes a single
        :class:`int` parameter ``n`` and returns ``n``
        colors. The maximum value of ``n`` varies
        depending on the parameters.

    Examples
    --------
    >>> palette = cmap_d_pal('viridis')
    >>> palette(5)
    ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725']
    """

    name: str

    def __post_init__(self):
        self._direction: Literal[1, -1] = 1
        # Accomodate matplotlib naming convention and allow
        # names that end with _r to return reversed colormaps
        if self.name.endswith("_r"):
            self._direction = -1
            self.name = self.name[:-2]
        self.cm = get_colormap(self.name)

    def __call__(self, n: int) -> Sequence[RGBHexColor]:
        return self.cm.discrete_palette(n)[:: self._direction]


class desaturate_pal(gradient_n_pal):
    """
    Create a palette that desaturate a color by some proportion

    Parameters
    ----------
    color : color
        html color name, hex, rgb-tuple
    prop : float
        saturation channel of color will be multiplied by
        this value
    reverse : bool
        Whether to reverse the palette.

    Returns
    -------
    out : function
        Continuous color palette that takes a single
        parameter either a :class:`float` or a sequence
        of floats maps those value(s) onto the palette
        and returns color(s). The float(s) must be
        in the range [0, 1].

    Examples
    --------
    >>> palette = desaturate_pal('red', .1)
    >>> palette([0, .25, .5, .75, 1])
    ['#ff0000', '#e21d1d', '#c53a3a', '#a95656', '#8c7373']
    """

    def __init__(self, color: str, prop: float, reverse: bool = False):
        if not 0 <= prop <= 1:
            raise ValueError("prop must be between 0 and 1")

        if isinstance(color, str):
            color = get_named_color(color)

        rgb = hex_to_rgb(color)
        h, l, s = colorsys.rgb_to_hls(*rgb)
        s *= prop
        desaturated_color = rgb_to_hex(colorsys.hls_to_rgb(h, l, s))
        colors = [color, desaturated_color]
        if reverse:
            colors = colors[::-1]

        super().__init__(colors)


@dataclass
class manual_pal(_discrete_pal):
    """
    Create a palette from a list of values

    Parameters
    ----------
    values : sequence
        Values that will be returned by the palette function.

    Returns
    -------
    out : function
        A function palette that takes a single
        :class:`int` parameter ``n`` and returns ``n`` values.

    Examples
    --------
    >>> palette = manual_pal(['a', 'b', 'c', 'd', 'e'])
    >>> palette(3)
    ['a', 'b', 'c']
    """

    values: Sequence[Any]

    def __post_init__(self):
        self.size = len(self.values)

    def __call__(self, n: int) -> Sequence[Any]:
        if n > self.size:
            warn(
                f"Palette can return a maximum of {self.size} values. "
                f"{n} values requested."
            )
        return self.values[:n]


def xkcd_palette(colors: Sequence[str]) -> Sequence[RGBHexColor]:
    """
    Make a palette with color names from the xkcd color survey.

    See xkcd for the full list of colors: http://xkcd.com/color/rgb/

    Parameters
    ----------
    colors : list of strings
        List of keys in the ``mizani.colors.xkcd_rgb`` dictionary.

    Returns
    -------
    palette : list
        List of colors as RGB hex strings.

    Examples
    --------
    >>> palette = xkcd_palette(['red', 'green', 'blue'])
    >>> palette
    ['#E50000', '#15B01A', '#0343DF']

    >>> from mizani._colors.named_colors import XKCD
    >>> list(sorted(XKCD.keys()))[:4]
    ['xkcd:acid green', 'xkcd:adobe', 'xkcd:algae', 'xkcd:algae green']
    """
    return [get_named_color(f"xkcd:{name}") for name in colors]


def crayon_palette(colors: Sequence[str]) -> Sequence[RGBHexColor]:
    """
    Make a palette with color names from Crayola crayons.

    The colors come from
    http://en.wikipedia.org/wiki/List_of_Crayola_crayon_colors

    Parameters
    ----------
    colors : list of strings
        List of keys in the ``mizani.colors.crayloax_rgb`` dictionary.

    Returns
    -------
    palette : list
        List of colors as RGB hex strings.

    Examples
    --------
    >>> palette = crayon_palette(['almond', 'silver', 'yellow'])
    >>> palette
    ['#EED9C4', '#C9C0BB', '#FBE870']

    >>> from mizani._colors.named_colors import CRAYON
    >>> list(sorted(CRAYON.keys()))[:3]
    ['crayon:almond', 'crayon:antique brass', 'crayon:apricot']
    """
    return [get_named_color(f"crayon:{name}") for name in colors]


@dataclass
class cubehelix_pal(_discrete_pal):
    """
    Utility for creating discrete palette from the cubehelix system.

    This produces a colormap with linearly-decreasing (or increasing)
    brightness. That means that information will be preserved if printed to
    black and white or viewed by someone who is colorblind.

    Parameters
    ----------
    start : float (0 <= start <= 3)
        The hue at the start of the helix.
    rot : float
        Rotations around the hue wheel over the range of the palette.
    gamma : float (0 <= gamma)
        Gamma factor to emphasize darker (gamma < 1) or lighter (gamma > 1)
        colors.
    hue : float (0 <= hue <= 1)
        Saturation of the colors.
    dark : float (0 <= dark <= 1)
        Intensity of the darkest color in the palette.
    light : float (0 <= light <= 1)
        Intensity of the lightest color in the palette.
    reverse : bool
        If True, the palette will go from dark to light.

    Returns
    -------
    out : function
        Continuous color palette that takes a single
        :class:`int` parameter ``n`` and returns ``n``
        equally spaced colors.


    References
    ----------
    Green, D. A. (2011). "A colour scheme for the display of astronomical
    intensity images". Bulletin of the Astromical Society of India, Vol. 39,
    p. 289-295.

    Examples
    --------
    >>> palette = cubehelix_pal()
    >>> palette(5)
    ['#edd1cb', '#d499a7', '#aa678f', '#6e4071', '#2d1e3e']
    """

    start: int = 0
    rotation: float = 0.4
    gamma: float = 1.0
    hue: float = 0.8
    light: float = 0.85
    dark: float = 0.15
    reverse: bool = False

    def __post_init__(self):
        self._chmap = CubeHelixMap(
            self.start,
            self.rotation,
            self.gamma,
            self.hue,
            self.light,
            self.dark,
            self.reverse,
        )

    def __call__(self, n: int) -> Sequence[RGBHexColor]:
        return self._chmap.discrete_palette(n)


def identity_pal() -> Callable[[], Any]:
    """
    Create palette that maps values onto themselves

    Returns
    -------
    out : function
        Palette function that takes a value or sequence of values
        and returns the same values.

    Examples
    --------
    >>> palette = identity_pal()
    >>> palette(9)
    9
    >>> palette([2, 4, 6])
    [2, 4, 6]
    """
    return identity
