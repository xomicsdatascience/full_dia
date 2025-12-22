import sys
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler

from full_dia.fdr import cal_q_pr_batch

sys.modules["cupy"] = mock.MagicMock()
sys.modules["torch"] = mock.MagicMock()


@pytest.fixture
def df_sample():
    n_peptides = 100
    n_score_cols = 200
    np.random.seed(0)

    # scores
    score_cols = {
        f"score_{i}": np.random.rand(n_peptides).astype(np.float32)
        for i in range(n_score_cols)
    }

    df = pd.DataFrame(score_cols)
    df["pr_id"] = [f"P{i % 3}" for i in range(n_peptides)]
    df["decoy"] = np.random.randint(0, 2, size=n_peptides)
    df["group_rank"] = 1
    df["score_big_deep_pre"] = np.random.rand(n_peptides).astype(np.float32)

    return df


def test_cal_q_pr_batch_first(df_sample):
    df_out, model, scaler = cal_q_pr_batch(
        df_sample, batch_size=10, n_model=2, model_trained=None, scaler=None
    )

    assert isinstance(df_out, pd.DataFrame)
    assert "cscore_pr_run" in df_out.columns
    assert "group_rank" in df_out.columns
    assert "q_pr_run" in df_out.columns

    assert np.all(df_out["cscore_pr_run"] <= 1.0)
    assert np.all(df_out["cscore_pr_run"] >= 0.0)
    assert np.all(df_out["q_pr_run"] <= 1.0)
    assert np.all(df_out["q_pr_run"] >= 0.0)

    assert isinstance(model, VotingClassifier)
    assert isinstance(scaler, StandardScaler)


def test_cal_q_pr_batch_second(df_sample):
    _, model, scaler = cal_q_pr_batch(
        df_sample, batch_size=10, n_model=2, model_trained=None, scaler=None
    )

    df_out, model2, scaler2 = cal_q_pr_batch(
        df_sample, batch_size=10, n_model=2, model_trained=model, scaler=scaler
    )

    assert isinstance(df_out, pd.DataFrame)
    assert "cscore_pr_run" in df_out.columns
    assert "group_rank" in df_out.columns
    assert "q_pr_run" in df_out.columns

    assert model2 == model
    assert scaler2 == scaler

    assert np.all(df_out["cscore_pr_run"] <= 1.0)
    assert np.all(df_out["cscore_pr_run"] >= 0.0)
    assert np.all(df_out["q_pr_run"] <= 1.0)
    assert np.all(df_out["q_pr_run"] >= 0.0)
