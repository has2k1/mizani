import numpy as np

from .. import ListedMap
from ..brewer import qualitative
from ..color_palette import palette

__all__ = (
    "Accent",
    "Dark2",
    "Paired",
    "Pastel1",
    "Pastel2",
    "Set1",
    "Set2",
    "Set3",
)


def _as_listed_map(name: str) -> ListedMap:
    p: palette = getattr(qualitative, name)
    colors = np.asarray(p.swatches[-1], dtype=float) / 255
    return ListedMap(colors)


Accent = _as_listed_map("Accent")
Dark2 = _as_listed_map("Dark2")
Paired = _as_listed_map("Paired")
Pastel1 = _as_listed_map("Pastel1")
Pastel2 = _as_listed_map("Pastel2")
Set1 = _as_listed_map("Set1")
Set2 = _as_listed_map("Set2")
Set3 = _as_listed_map("Set3")
