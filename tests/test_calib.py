import sys
from unittest import mock

import numpy as np
import pandas as pd

from full_dia import calib, cfg

sys.modules["cupy"] = mock.MagicMock()
sys.modules["torch"] = mock.MagicMock()


def test_calib_rt():
    cfg.load_default()

    n = 500
    df_seed = pd.DataFrame(
        {
            "simple_seq": [f"PEP{i}" for i in range(n)],
            "locus": np.arange(n),
            "score_deep": np.linspace(0, 1, n),
            "pred_irt": np.linspace(0, 100, n),
            "measure_rt": np.linspace(5, 105, n) + np.random.normal(0, 0.3, n),
            "measure_pr_mz": np.full(n, 500.0),
            "measure_im": np.full(n, 1.0),
        }
    )
    cfg.tol_rt = df_seed["measure_rt"].max() / 10
    df_lib = df_seed.copy()

    out_seed, out_lib = calib.calib_rt(df_seed, df_lib)

    assert isinstance(out_seed, pd.DataFrame)
    assert isinstance(out_lib, pd.DataFrame)

    assert "pred_rt" in out_seed.columns
    assert "bias_rt" in out_seed.columns
    assert "pred_rt" in out_lib.columns

    # row count sanity
    assert 0 < len(out_seed) <= len(df_seed)
    # RT validity check
    assert np.all(out_lib["pred_rt"].values >= 0)
    # iRT → RT monotonicity
    corr = np.corrcoef(out_seed["pred_irt"], out_seed["pred_rt"])[0, 1]
    assert corr > 0.9


def test_calib_im():
    cfg.load_default()

    n = 500
    df_seed = pd.DataFrame(
        {
            "score_deep": np.linspace(10, 100, n),
            "pred_iim": np.linspace(0.5, 1, n),
            "pred_im": np.linspace(0.5, 1, n) + np.random.normal(0, 0.05, n),
            "measure_im": np.linspace(0.5, 1, n) + np.random.normal(0, 0.05, n),
        }
    )
    df_lib = df_seed.copy()

    out_tol, out_lib = calib.calib_im(df_seed, df_lib)

    assert isinstance(out_tol, pd.DataFrame)
    assert isinstance(out_lib, pd.DataFrame)
    assert "pred_im" in out_tol.columns
    assert "bias_im" in out_tol.columns
    assert "pred_im" in out_lib.columns

    # row count sanity
    assert 0 < len(out_tol) <= len(df_seed)
    # IM validity check
    assert np.all(out_lib["pred_im"].values >= 0)
    # pred_iim → pred_im correlation
    corr = np.corrcoef(out_tol["pred_iim"], out_tol["pred_im"])[0, 1]
    assert corr > 0.9


class DummyTims:
    """Minimal dummy Tims object for testing update_info_mz"""

    def __init__(self, n_swath, n_points):
        self.d_ms1_maps = []
        self.d_ms2_maps = []
        for _ in range(n_swath):
            dummy_map = (
                np.arange(n_points),  # all_rt
                np.ones(n_points),  # cycle_valid_lens
                np.arange(n_points),  # all_push
                np.linspace(100, 200, n_points),  # all_tof
                np.random.rand(n_points),  # all_height
                np.ones(n_points),  # cycle_valid_lens2
                np.arange(n_points),  # all_push2
                np.linspace(100, 200, n_points),  # all_tof2
                np.random.rand(n_points),  # all_height2
            )
            self.d_ms1_maps.append(dummy_map)
            self.d_ms2_maps.append(dummy_map)

    def get_dia_quadrupole(self):
        return range(len(self.d_ms1_maps))


def test_calib_mz():
    cfg.load_default()

    n = 20
    df_seed = pd.DataFrame(
        {
            "score_deep": np.linspace(0, 1, n),
            "measure_pr_mz": np.linspace(500, 1500, n),
            "pr_mz": np.linspace(500, 1500, n) + np.random.normal(0, 0.1, n),
        }
    )
    ms = DummyTims(n_swath=10, n_points=1000)

    calib.calib_mz(df_seed, ms)

    # check df_seed unchanged columns
    for col in ["score_deep", "measure_pr_mz", "pr_mz"]:
        assert col in df_seed.columns

    # check ms.d_ms1_maps / ms.d_ms2_maps updated
    for swath_id in range(1, len(ms.d_ms1_maps)):
        all_tof = ms.d_ms1_maps[swath_id][3]
        all_tof2 = ms.d_ms1_maps[swath_id][7]
        assert isinstance(all_tof, np.ndarray)
        assert all_tof.dtype == np.float32
        assert isinstance(all_tof2, np.ndarray)
        assert all_tof2.dtype == np.float32
    for swath_id in range(1, len(ms.d_ms2_maps)):
        all_tof = ms.d_ms2_maps[swath_id][3]
        all_tof2 = ms.d_ms2_maps[swath_id][7]
        assert isinstance(all_tof, np.ndarray)
        assert all_tof.dtype == np.float32
        assert isinstance(all_tof2, np.ndarray)
        assert all_tof2.dtype == np.float32
