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
import typing
from dataclasses import dataclass
from warnings import warn

import numpy as np

from .bounds import rescale
from .colors import crayon_rgb, hsluv, rgb_to_hex, xkcd_rgb
from .utils import identity

if typing.TYPE_CHECKING:
    from typing import Any, Literal, Optional, Sequence

    from matplotlib.colors import Colormap

    from mizani.typing import (
        Callable,
        ColorScheme,
        ColorSchemeShort,
        FloatVector,
        NumVector,
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


class _discrete_pal:
    """
    Discrete palette maker
    """

    def __call__(self, n: int) -> Sequence[Any]:
        """
        Palette method
        """
        ...


class _continuous_pal:
    """
    Continuous palette maker
    """

    def __call__(self, x: Sequence[Any]) -> FloatVector:
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

    def __call__(self, x: FloatVector) -> FloatVector:
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

    def __call__(self, x: FloatVector) -> FloatVector:
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

    def __call__(self, x: NumVector) -> NumVector:
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
    ['#333333', '#737373', '#989898', '#b5b5b5', '#cccccc']
    """

    start: float = 0.2
    end: float = 0.8

    def __call__(self, n: int) -> Sequence[RGBHexColor | None]:
        from matplotlib.colors import LinearSegmentedColormap

        gamma = 2.2
        ends = ((0.0, self.start, self.start), (1.0, self.end, self.end))
        cdict = {"red": ends, "green": ends, "blue": ends}
        grey_cmap = LinearSegmentedColormap("grey", cdict)

        # The grey scale points are linearly separated in
        # gamma encoded space
        x = np.linspace(self.start**gamma, self.end**gamma, n)
        # Map points onto the [0, 1] palette domain
        vals = (x ** (1.0 / gamma) - self.start) / (self.end - self.start)
        return ratios_to_colors(vals, grey_cmap)


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


def brewer_pal(
    type: ColorScheme | ColorSchemeShort = "seq",
    palette: int = 1,
    direction: Literal[1, -1] = 1,
):
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

        from mizani.colors.brewer import get_palette_names

        print(get_palette_names("sequential"))
        print(get_palette_names("qualitative"))
        print(get_palette_names("diverging"))
    """
    from .colors import brewer

    if direction != 1 and direction != -1:
        raise ValueError("direction should be 1 or -1.")

    pal = brewer.get_color_palette(type, palette)

    def _brewer_pal(n):
        # Only draw the maximum allowable colors from the palette
        # and fill any remaining spots with None
        _n = min(max(n, pal.min_colors), pal.max_colors)
        color_map = pal.get_hex_swatch(_n)
        colors = color_map[:n]
        if n > pal.max_colors:
            msg = (
                "Warning message:"
                f"Brewer palette {pal.name} has a maximum "
                f"of {pal.max_colors} colors Returning the palette you "
                "asked for with that many colors"
            )
            warn(msg)
            colors = colors + [None] * (n - pal.max_colors)
        return colors[::direction]

    return _brewer_pal


def ratios_to_colors(
    values: FloatVector, colormap: "Colormap"
) -> Sequence[RGBHexColor | None]:
    """
    Map values in the range [0, 1] onto colors

    Parameters
    ----------
    values : array_like
        Numeric(s) in the range [0, 1]
    colormap : cmap
        Matplotlib colormap to use for the mapping

    Returns
    -------
    out : list
        Color(s) corresponding to the values
    """
    color_tuples = colormap(values)
    hex_colors = [rgb_to_hex(t) for t in color_tuples]

    nan_bool_idx = np.isnan(values) | np.isinf(values)
    if nan_bool_idx.any():
        hex_colors = [
            None if isnan else color
            for color, isnan in zip(hex_colors, nan_bool_idx)
        ]
    return hex_colors


@dataclass
class gradient_n_pal(_continuous_pal):
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
    name : str
        Name to call the resultant MPL colormap

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
    ['#ff0000', '#bf0040', '#7f0080', '#3f00c0', '#0000ff']
    >>> palette([-np.inf, 0, np.nan, 1, np.inf])
    [None, '#ff0000', None, '#0000ff', None]
    """

    colors: Sequence[str]
    values: Optional[Sequence[float]] = None
    name: str = "gradientn"

    def __post_init__(self):
        from matplotlib.colors import LinearSegmentedColormap

        # Note: For better results across devices and media types,
        # it would be better to do the interpolation in
        # Lab color space.
        if self.values is None:
            self.colormap = LinearSegmentedColormap.from_list(
                self.name, self.colors
            )
        else:
            self.colormap = LinearSegmentedColormap.from_list(
                self.name, list(zip(self.values, self.colors))
            )

    def __call__(self, x: FloatVector) -> Sequence[RGBHexColor | None]:
        return ratios_to_colors(x, self.colormap)


@dataclass
class cmap_pal(_continuous_pal):
    """
    Create a continuous palette using an MPL colormap

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
        import matplotlib as mpl

        self.colormap = mpl.colormaps[self.name]

    def __call__(self, x: FloatVector) -> Sequence[RGBHexColor | None]:
        return ratios_to_colors(x, self.colormap)


@dataclass
class cmap_d_pal(_discrete_pal):
    """
    Create a discrete palette using an MPL Listed colormap

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
    ['#440154', '#3b528b', '#21918c', '#5cc863', '#fde725']
    """

    name: str

    def __post_init__(self):
        import matplotlib as mpl
        from matplotlib.colors import ListedColormap

        self.colormap = mpl.colormaps[self.name]

        if not isinstance(self.colormap, ListedColormap):
            raise ValueError(
                "For a discrete palette, cmap must be of type "
                "matplotlib.colors.ListedColormap"
            )

        self.num_colors = len(self.colormap.colors)

    def __call__(self, n: int) -> Sequence[RGBHexColor]:
        if n > self.num_colors:
            raise ValueError(
                f"cmap {self.name} has {self.num_colors} colors you "
                f"requested {n} colors."
            )
        if self.num_colors < 256:
            return [rgb_to_hex(c) for c in self.colormap.colors[:n]]
        else:
            # Assume these are continuous and get colors equally spaced
            # intervals  e.g. viridis is defined with 256 colors
            idx = np.linspace(0, self.num_colors - 1, n).round().astype(int)
            return [rgb_to_hex(self.colormap.colors[i]) for i in idx]


class desaturate_pal(gradient_n_pal):
    """
    Create a palette that desaturate a color by some proportion

    Parameters
    ----------
    color : matplotlib color
        hex, rgb-tuple, or html color name
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
        from matplotlib.colors import colorConverter

        if not 0 <= prop <= 1:
            raise ValueError("prop must be between 0 and 1")

        rgb = colorConverter.to_rgb(color)
        h, l, s = colorsys.rgb_to_hls(*rgb)
        s *= prop
        desaturated_color = colorsys.hls_to_rgb(h, l, s)
        colors = [color, desaturated_color]
        if reverse:
            colors = colors[::-1]

        super().__init__(colors, name="desaturated")


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
    ['#e50000', '#15b01a', '#0343df']

    >>> from mizani.colors import xkcd_rgb
    >>> list(sorted(xkcd_rgb.keys()))[:5]
    ['acid green', 'adobe', 'algae', 'algae green', 'almost black']
    """
    return [xkcd_rgb[name] for name in colors]


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
    ['#eed9c4', '#c9c0bb', '#fbe870']

    >>> from mizani.colors import crayon_rgb
    >>> list(sorted(crayon_rgb.keys()))[:5]
    ['almond', 'antique brass', 'apricot', 'aquamarine', 'asparagus']
    """
    return [crayon_rgb[name] for name in colors]


@dataclass
class cubehelix_pal(_continuous_pal):
    """
    Utility for creating continuous palette from the cubehelix system.

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
    ['#edd1cb', '#d499a7', '#aa688f', '#6e4071', '#2d1e3e']
    """

    start: int = 0
    rot: float = 0.4
    gamma: float = 1.0
    hue: float = 0.8
    light: float = 0.85
    dark: float = 0.15
    reverse: bool = False

    def __post_init__(self):
        from matplotlib._cm import cubehelix
        from matplotlib.colors import LinearSegmentedColormap

        cdict = cubehelix(self.gamma, self.start, self.rot, self.hue)
        self.cubehelix_cmap = LinearSegmentedColormap("cubehelix", cdict)

    def __call__(self, n: int) -> Sequence[RGBHexColor]:
        values = np.linspace(self.light, self.dark, n)
        return [rgb_to_hex(self.cubehelix_cmap(x)) for x in values]


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
