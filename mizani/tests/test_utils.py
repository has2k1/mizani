from __future__ import division

import numpy.testing as npt
import pytest


from ..utils import (seq, fullseq, round_any, min_max, match,
                     precision)


def test_seq():
    result = seq(4.1, 5.2, 0.1)
    npt.assert_approx_equal(result[-1], 5.2)

    result = seq(1, 10, length_out=10)
    npt.assert_array_equal(result, range(1, 11))

    with pytest.raises(ValueError):
        seq(1, 10, length_out=0)


def test_fullseq():
    result = fullseq((1, 3), size=.5)
    npt.assert_array_equal(result, [1, 1.5, 2, 2.5, 3])

    result = fullseq((1, 3.2), size=.5)
    npt.assert_array_equal(result, [1, 1.5, 2, 2.5, 3, 3.5])

    result = fullseq((0.8, 3), size=.5)
    npt.assert_array_equal(result, [0.5, 1, 1.5, 2, 2.5, 3])

    result = fullseq((1, 3), size=.5, pad=True)
    npt.assert_array_equal(result, [0.5, 1, 1.5, 2, 2.5, 3, 3.5])

    result = fullseq((2, 2), size=1)
    npt.assert_array_equal(result, [1.5, 2.5])


def test_round_any():
    x = 4.632
    assert round_any(x, 1) == 5
    assert round_any(x, 2) == 4
    assert round_any(x, 3) == 6
    assert round_any(x, 4) == 4
    assert round_any(x, 5) == 5
    assert round_any(x, 1.5) == 4.5


def test_min_max():
    x = [1, 2, 3, 4, 5]
    _min, _max = min_max(x)
    assert _min == 1
    assert _max == 5

    x = [1, float('-inf'), 3, 4, 5]
    _min, _max = min_max(x)
    assert _min == 1
    assert _max == 5

    _min, _max = min_max(x, finite=False)
    assert _min == float('-inf')
    assert _max == 5

    x = [1, 2, float('nan'), 4, 5]
    _min, _max = min_max(x, nan_rm=True)
    assert _min == 1
    assert _max == 5

    x = [1, 2, float('nan'), 4, 5, float('inf')]
    _min, _max = min_max(x, nan_rm=True, finite=False)
    assert _min == 1
    assert _max == float('inf')

    _min, _max = min_max(x)
    assert str(_min) == 'nan'
    assert str(_max) == 'nan'


def test_match():
    v1 = [0, 1, 2, 3, 4, 5]
    v2 = [5, 4, 3, 2, 1, 0]
    result = match(v1, v2)
    assert result == v2

    # Positions of the first match
    result = match(v1, v2+v2)
    assert result == v2

    result = match(v1, v2, incomparables=[1, 2])
    assert result == [5, -1, -1, 2, 1, 0]

    result = match(v1, v2, start=1)
    assert result == [6, 5, 4, 3, 2, 1]

    v2 = [5, 99, 3, 2, 1, 0]
    result = match(v1, v2)
    assert result == [5, 4, 3, 2, -1, 0]


def test_precision():
    assert precision(0.0037) == .001
    assert precision(0.5) == .1
    assert precision(9) == 1
    assert precision(24) == 10
    assert precision(784) == 100
    assert precision([0, 0]) == 1
