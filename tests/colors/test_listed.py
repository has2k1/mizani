import numpy as np
import pytest

from mizani.colors import ListedMap


def test_continuous():
    gmap = ListedMap(["blue", "red", "green"])
    x = np.linspace(0, 1, 10)
    colors = gmap.continuous_palette(x)
    assert len(colors) == 10
    assert [c is not None and c.startswith("#") for c in colors]
