from mizani.colors import CubeHelixMap


def test_continuous():
    x = [0.1, 0.2, 0.3, 0.4, 0.5]
    chmap = CubeHelixMap(reverse=True)
    colors = chmap.continuous_palette(x)
    assert len(colors) == 5
    assert all(c is not None and c.startswith("#") for c in colors)

    x = [0.1, 0.2, float("nan"), 0.4, float("inf")]
    colors = chmap.continuous_palette(x)
    assert len(colors) == 5
    assert colors[2] is None
    assert colors[4] is None
    assert all(colors[i].startswith("#") for i in (0, 1, 3))
