import sys
from unittest import mock

import numpy as np

from full_dia import utils

sys.modules["cupy"] = mock.MagicMock()
sys.modules["torch"] = mock.MagicMock()


def test_cal_group_rank():
    x = np.array(
        [
            0.2,  # group 0: no tie
            0.5,
            0.1,
            1.0,  # group 1: single
            0.9,
            0.9,
            0.3,  # group 2: tie
        ],
        dtype=np.float32,
    )
    group_size_cumsum = np.array([0, 3, 4, 7], dtype=np.int64)
    rank = utils.cal_group_rank(x, group_size_cumsum)
    # group 0: 0.5 > 0.2 > 0.1
    assert rank[0:3].tolist() == [2, 1, 3]
    # group 1
    assert rank[3] == 1
    # group 2: two 0.9 tied → should share top ranks {1,2}
    tied_ranks = sorted(rank[4:6].tolist())
    assert tied_ranks == [1, 2]
    assert rank[6] == 3  # 0.3 lowest


def test_move_all_zeros_end():
    a = np.array(
        [
            [1, 0, 2, 0],
            [0, 0, 3, 4],
            [5, 6, 7, 8],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    out = utils.move_all_zeros_end(a.copy())
    expected = np.array(
        [
            [1, 2, 0, 0],
            [3, 4, 0, 0],
            [5, 6, 7, 8],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    assert np.array_equal(out, expected)
