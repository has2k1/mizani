from mizani._colors import SegmentFunctionMap
from mizani._colors._colormaps._maps._segment_function import GPF, _gpf_32


def test_discrete():
    cmap = SegmentFunctionMap(
        {"red": _gpf_32, "green": GPF[3], "blue": GPF[17]}
    )
    colors = cmap.discrete_palette(5)
    assert len(colors) == 5
    assert [c[0] == "#" and len(c) == 7 for c in colors]
