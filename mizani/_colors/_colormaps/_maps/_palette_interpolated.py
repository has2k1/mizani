from mizani._colors._palettes.brewer import (
    diverging,
    qualitative,
    sequential,
)
from mizani._colors._palettes.other import (
    qualitative as other_qualitative,
)

__all__ = (
    # diverging
    "BrBG",
    "PiYG",
    "PRGn",
    "PuOr",
    "RdBu",
    "RdGy",
    "RdYlBu",
    "RdYlGn",
    "Spectral",
    # qualitative
    "Accent",
    "Dark2",
    "Paired",
    "Pastel1",
    "Pastel2",
    "Set1",
    "Set2",
    "Set3",
    # sequential
    "Blues",
    "BuGn",
    "BuPu",
    "GnBu",
    "Greens",
    "Greys",
    "Oranges",
    "OrRd",
    "PuBu",
    "PuBuGn",
    "PuRd",
    "Purples",
    "RdPu",
    "Reds",
    "YlGn",
    "YlGnBu",
    "YlOrBr",
    "YlOrRd",
    # qualitative
    "category10",
    "category20",
    "category20b",
    "category20c",
    "observable10",
    "tableau10",
    "tableau20",
)

BrBG = diverging.BrBG.colormap
PiYG = diverging.PiYG.colormap
PRGn = diverging.PRGn.colormap
PuOr = diverging.PuOr.colormap
RdBu = diverging.RdBu.colormap
RdGy = diverging.RdGy.colormap
RdYlBu = diverging.RdYlBu.colormap
RdYlGn = diverging.RdYlGn.colormap
Spectral = diverging.Spectral.colormap

Accent = qualitative.Accent.colormap
Dark2 = qualitative.Dark2.colormap
Paired = qualitative.Paired.colormap
Pastel1 = qualitative.Pastel1.colormap
Pastel2 = qualitative.Pastel2.colormap
Set1 = qualitative.Set1.colormap
Set2 = qualitative.Set2.colormap
Set3 = qualitative.Set3.colormap

Blues = sequential.Blues.colormap
BuGn = sequential.BuGn.colormap
BuPu = sequential.BuPu.colormap
GnBu = sequential.GnBu.colormap
Greens = sequential.Greens.colormap
Greys = sequential.Greys.colormap
Oranges = sequential.Oranges.colormap
OrRd = sequential.OrRd.colormap
PuBu = sequential.PuBu.colormap
PuBuGn = sequential.PuBuGn.colormap
PuRd = sequential.PuRd.colormap
Purples = sequential.Purples.colormap
RdPu = sequential.RdPu.colormap
Reds = sequential.Reds.colormap
YlGn = sequential.YlGn.colormap
YlGnBu = sequential.YlGnBu.colormap
YlOrBr = sequential.YlOrBr.colormap
YlOrRd = sequential.YlOrRd.colormap

category10 = other_qualitative.category10.colormap
category20 = other_qualitative.category20.colormap
category20b = other_qualitative.category20b.colormap
category20c = other_qualitative.category20c.colormap
observable10 = other_qualitative.observable10.colormap
tableau10 = other_qualitative.tableau10.colormap
tableau20 = other_qualitative.tableau20.colormap
