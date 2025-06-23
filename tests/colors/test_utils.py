import pandas as pd
import pytest

from mizani._colors.utils import color_tuple_to_hex, to_rgba


def test_to_rgba():
    assert to_rgba("red", 0.5) == "#FF000080"
    assert to_rgba("red", 1) == "#FF0000FF"
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

    assert to_rgba((0, 0, 1, 0.2), 1) == (0, 0, 1, 0.2)
    assert to_rgba("none", 0.5) == "none"

    with pytest.raises(ValueError):
        to_rgba("red", "0")  # pyright: ignore[reportCallIssue,reportArgumentType]

    with pytest.raises(ValueError):
        to_rgba((1, 0, 0), "0")  # pyright: ignore[reportCallIssue,reportArgumentType]

    with pytest.raises(TypeError):
        to_rgba((0, 0, 1, 0.2, 0.2), 1)  # pyright: ignore[reportCallIssue,reportArgumentType]


def test_color_tuple_to_hex():
    assert color_tuple_to_hex((1, 0, 0)) == "#FF0000"
    assert color_tuple_to_hex((1, 0, 0, 0.5)) == "#FF000080"
