import sys
from unittest import mock

import numpy as np
import pandas as pd

from full_dia import assemble

sys.modules["cupy"] = mock.MagicMock()
sys.modules["torch"] = mock.MagicMock()


def test_assemble():
    df = pd.DataFrame()
    df["strip_seq"] = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    df["protein_id"] = [
        "P7",
        "P4;P6;P9;P10",
        "P1",
        "P1;P5",
        "P7",
        "P3;P6",
        "P1",
        "P1;P2;P5;P8",
        "P1",
        "P4;P9;P10",
    ]
    df["q_pr_run"] = 0
    df["cscore_pr_run"] = np.linspace(0, 1, 10)  # for a tie
    df = assemble.assemble_pep_to_pg(df, 0.01, "run")

    expects = ["P7", "P10;P9;P4", "P1", "P1", "P7", "P3", "P1", "P1", "P1", "P10;P9;P4"]

    for actual, expect in zip(df["protein_group"], expects):
        actual_set = set(actual.split(";"))
        expect_set = set(expect.split(";"))
        assert actual_set == expect_set
