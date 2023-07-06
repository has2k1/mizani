import json
from pathlib import Path

import pytest
from numpy.testing import assert_allclose

from mizani.colors import hsluv

ATOL = 1e-11


def test_rgb_range():
    step = 10
    for h in range(0, 361, step):
        for s in range(0, 101, step):
            for l in range(0, 101, step):
                rgb = hsluv.hsluv_to_rgb((h, s, l))
                for c in rgb:
                    assert -ATOL < c < 1 + ATOL

                rgb = hsluv.hpluv_to_rgb((h, s, l))
                for c in rgb:
                    assert -ATOL < c < 1 + ATOL


@pytest.mark.slow
def test_snapshort():
    # Load snapshot into memory
    filename = Path(__file__).parent / "data/hsluv-snapshot-rev4.json"
    with open(filename) as f:
        snapshot = json.load(f)

    for hex_color, colors in snapshot.items():
        # Test forward functions
        test_rgb = hsluv.hex_to_rgb(hex_color)
        assert_close(test_rgb, colors["rgb"])
        test_xyz = hsluv.rgb_to_xyz(test_rgb)
        assert_close(test_xyz, colors["xyz"])
        test_luv = hsluv.xyz_to_luv(test_xyz)
        assert_close(test_luv, colors["luv"])
        test_lch = hsluv.luv_to_lch(test_luv)
        assert_close(test_lch, colors["lch"])
        test_hsluv = hsluv.lch_to_hsluv(test_lch)
        assert_close(test_hsluv, colors["hsluv"])
        test_hpluv = hsluv.lch_to_hpluv(test_lch)
        assert_close(test_hpluv, colors["hpluv"])

        # Test backward functions
        test_lch = hsluv.hsluv_to_lch(colors["hsluv"])
        assert_close(test_lch, colors["lch"])
        test_lch = hsluv.hpluv_to_lch(colors["hpluv"])
        assert_close(test_lch, colors["lch"])
        test_luv = hsluv.lch_to_luv(test_lch)
        assert_close(test_luv, colors["luv"])
        test_xyz = hsluv.luv_to_xyz(test_luv)
        assert_close(test_xyz, colors["xyz"])
        test_rgb = hsluv.xyz_to_rgb(test_xyz)
        assert_close(test_rgb, colors["rgb"])
        assert hsluv.rgb_to_hex(test_rgb) == hex_color

        # Full test
        assert hsluv.hsluv_to_hex(colors["hsluv"]) == hex_color
        assert_close(hsluv.hex_to_hsluv(hex_color), colors["hsluv"])
        assert hsluv.hpluv_to_hex(colors["hpluv"]) == hex_color
        assert_close(hsluv.hex_to_hpluv(hex_color), colors["hpluv"])


def assert_close(t1, t2):
    assert_allclose(t1, t2, atol=ATOL)
