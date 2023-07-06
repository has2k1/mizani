import pytest

from mizani.colors import GradientMap


def test_number_of_values():
    with pytest.raises(ValueError):
        GradientMap(["blue", "red", "green"], [0.0, 0.2, 0.9, 1.0])

    with pytest.raises(ValueError):
        GradientMap(["blue", "red", "green"], [0, 0.2, 0.9])

    with pytest.raises(ValueError):
        GradientMap(["blue"], [0])


def test_discrete():
    gmap = GradientMap(["blue", "red", "green"], [0, 0.5, 1])
    colors = gmap.discrete_palette(5)
    assert len(colors) == 5
    assert [c is not None and c.startswith("#") for c in colors]
