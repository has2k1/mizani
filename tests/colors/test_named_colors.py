import pytest

from mizani._colors import get_named_color


def test_bad_name():
    with pytest.raises(ValueError):
        get_named_color("tada")
