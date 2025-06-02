import pandas as pd

from mizani._colors.utils import to_rgba


def test_to_rgba():
    assert to_rgba("red", 0.5) == "#FF000080"
    assert to_rgba("#FF0000", 0.5) == "#FF000080"
    assert to_rgba("#FF000022", 0.5) == "#FF000022"
    assert to_rgba((0, 1, 0), 0.5) == (0, 1, 0, 0.5)
    assert to_rgba((0, 1, 0, 0.5), 0.9) == (0, 1, 0, 0.5)
    x = ["red", "green", "blue"]
    assert to_rgba(x, 0.5) == ["#FF000080", "#00800080", "#0000FF80"]
    assert to_rgba(x, (0.4, 0.5, 0.6)) == [
        "#FF000066",
        "#00800080",
        "#0000FF99",
    ]
    assert to_rgba(pd.Series(x), (0.4, 0.5, 0.6)) == [
        "#FF000066",
        "#00800080",
        "#0000FF99",
    ]
