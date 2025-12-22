import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from full_dia import cfg, utils
from full_dia.log import Logger

os.environ["PYTHONWARNINGS"] = "ignore"  # multiprocess
warnings.simplefilter("ignore")
warnings.filterwarnings(action="ignore", category=UserWarning)
warnings.filterwarnings(action="ignore", category=ConvergenceWarning)

try:
    _ = profile
except NameError:

    def profile(func):
        return func


logger = Logger.get_logger()


def adjust_rubbish_q(df: pd.DataFrame, batch_num: int) -> None:
    """
    If the #ids are less than 5000, then we can set the rubbish_q_cut to 0.75 to save more #ids.
    """
    ids = df[
        (df["q_pr_run"] < 0.01) & (df["decoy"] == 0) & (df["group_rank"] == 1)
    ].pr_id.nunique()
    ids = ids * batch_num
    if ids < 5000:
        cfg.rubbish_q_cut = 0.75
    else:
        cfg.rubbish_q_cut = cfg.rubbish_q_cut


def cal_q_pr_core(df: pd.DataFrame, run_or_global: str):
    """
    The core function to calculate the q values of peps using DIA-NN's model.

    Parameters
    ----------
    df : pd.DataFrame
        Provide "cscore" values of peptides.

    run_or_global : str
        Specify the calculation on "run" or "global" level.

    Returns
    -------
    df : pd.DataFrame
        Add a new column: 'q_pr'.
    """
    col_score = "cscore_pr_" + run_or_global
    col_out = "q_pr_" + run_or_global

    df = df.sort_values(by=col_score, ascending=False, ignore_index=True)
    target_num = (df.decoy == 0).cumsum()
    decoy_num = (df.decoy == 1).cumsum()

    target_num[target_num == 0] = 1
    df[col_out] = decoy_num / target_num

    df[col_out] = df[col_out][::-1].cummin()
    return df


@profile
def cal_q_pr_batch(
    df: pd.DataFrame,
    batch_size: int,
    n_model: int,
    model_trained: list[MLPClassifier] | None = None,
    scaler: StandardScaler | None = None,
) -> tuple:
    """
    Calculate the q values of peptides in a batch.

    Parameters
    ----------
    df : pd.DataFrame
        Provide columns: "pr_id", "score_", "decoy".

    batch_size : int
        The batch size for training an MLP.

    n_model : int
        The number of models to ensemble.

    model_trained : list[MLPClassifier], default=None
        The trained ensemble models or None.

    scaler : StandardScaler, default=None
        The trained scaler or None.

    Returns
    -------
    tuple
        df : pd.DataFrame
            Add new columns: "cscore_pr_run", "group_rank", "q_pr_run"

        model_trained : list[MLPClassifier]
            The trained ensemble models.

        scaler : StandardScaler
            The trained scaler.
    """
    col_idx = df.columns.str.startswith("score_")
    # assert sum(col_idx) == 392
    # logger.info('cols num: {}'.format(sum(col_idx)))

    X = df.loc[:, col_idx].values
    assert X.dtype == np.float32
    y = 1 - df["decoy"].values  # targets is positives
    if scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)  # no scale to Tree models
    else:
        X = scaler.transform(X)

    # train
    if model_trained is None:  # the first batch
        decoy_deeps = df.loc[df["decoy"] == 1, "score_big_deep_pre"].values
        decoy_m, decoy_u = np.mean(decoy_deeps), np.std(decoy_deeps)
        good_cut = min(0.5, decoy_m + 1.5 * decoy_u)
        logger.info(f"Training with big_score_cut: {good_cut:.2f}")
        train_idx = (df["group_rank"] == 1) & (df["score_big_deep_pre"] > good_cut)
        X_train = X[train_idx]
        y_train = y[train_idx]

        n_pos, n_neg = sum(y_train == 1), sum(y_train == 0)
        info = "Training the NN model: {} pos, {} neg".format(n_pos, n_neg)
        logger.info(info)

        param = (25, 20, 15, 10, 5)
        mlps = [
            MLPClassifier(
                max_iter=1,
                shuffle=True,
                random_state=i,  # init weights and shuffle
                learning_rate_init=0.003,
                solver="adam",
                batch_size=batch_size,  # DIA-NN is 50
                activation="relu",
                hidden_layer_sizes=param,
            )
            for i in range(n_model)
        ]
        names = [f"mlp{i}" for i in range(n_model)]
        model = VotingClassifier(
            estimators=list(zip(names, mlps)),
            voting="soft",
            n_jobs=1 if __debug__ else n_model,
        )
        model.fit(X_train, y_train)

        n_pos, n_neg = sum(y == 1), sum(y == 0)
        info = "Predicting by the NN model: {} pos, {} neg".format(n_pos, n_neg)
        logger.info(info)
        cscore = model.predict_proba(X)[:, 1]
    else:
        model = model_trained
        n_pos, n_neg = sum(y == 1), sum(y == 0)
        info = "Predicting by the NN model: {} pos, {} neg".format(n_pos, n_neg)
        logger.info(info)
        cscore = model.predict_proba(X)[:, 1]

    df["cscore_pr_run"] = cscore

    # group rank
    group_size = df.groupby("pr_id", sort=False).size()
    group_size_cumsum = np.concatenate([[0], np.cumsum(group_size)])
    group_rank = utils.cal_group_rank(df["cscore_pr_run"].values, group_size_cumsum)
    df["group_rank"] = group_rank
    df = df.loc[group_rank == 1]

    df = cal_q_pr_core(df, "run")

    return df, model, scaler


@profile
def cal_q_pr_combined(df: pd.DataFrame, batch_size: int, n_model: int) -> pd.DataFrame:
    """
    Calculate the q values of peptides across the batches using DIA-NN's model.

    Parameters
    ----------
    df : pd.DataFrame
        Provide columns: "pr_id", "score_", "decoy".

    batch_size : int
        The batch size for training an MLP.

    n_model : int
        The number of models to ensemble.

    Returns
    -------
    df : pd.DataFrame
        Add new columns: "cscore_pr_run", "q_pr_run"
    """
    col_idx = df.columns.str.startswith("score_")
    logger.info("scores items: {}".format(sum(col_idx)))

    X = df.loc[:, col_idx].values
    assert X.dtype == np.float32
    y = 1 - df["decoy"].values  # targets is positives

    # select by train_nn_type: 1-hard, 2-easy, 3-cross
    scaler = StandardScaler()
    X = scaler.fit_transform(X)  # no scale to Tree models

    # train
    n_pos, n_neg = sum(y == 1), sum(y == 0)
    info = "Training the NN model: {} pos, {} neg".format(n_pos, n_neg)
    logger.info(info)

    param = (25, 20, 15, 10, 5)
    mlps = [
        MLPClassifier(
            max_iter=1,
            shuffle=True,
            random_state=i,  # init weights and shuffle
            learning_rate_init=0.003,
            solver="adam",
            batch_size=batch_size,  # DIA-NN is 50
            activation="relu",
            hidden_layer_sizes=param,
        )
        for i in range(n_model)
    ]
    names = [f"mlp{i}" for i in range(n_model)]
    model = VotingClassifier(
        estimators=list(zip(names, mlps)),
        voting="soft",
        n_jobs=1 if __debug__ else n_model,
    )
    model.fit(X, y)

    # pred
    cscore = model.predict_proba(X)[:, 1]
    df["cscore_pr_run"] = cscore

    # mirrors does not involve this
    df = cal_q_pr_core(df, "run")
    return df


def cal_q_pg(
    df_input_raw: pd.DataFrame, q_pr_cut: float, run_or_global: str
) -> pd.DataFrame:
    """
    Calculate the q values of pgs based on the assigned peptides.

    Parameters
    ----------
    df_input_raw : pd.DataFrame
        Provide columns: "strip_seq", "cscore_pr", "decoy", "q_pr".

    q_pr_cut : float
        The q value cut to select which peps will be used to calculate the cscore of a pg.

    run_or_global : str
        Specify the calculation on "run" or "global" level.

    Returns
    -------
    df_res : pd.DataFrame
        Add new columns: "cscore_pg", "q_pg"
    """
    x = run_or_global
    df_na = df_input_raw[df_input_raw["cscore_pr_" + x].isna()]
    df_input = df_input_raw[~df_input_raw["cscore_pr_" + x].isna()]

    if "strip_seq" not in df_input.columns:
        if "simple_seq" not in df_input.columns:
            df_input["simple_seq"] = (
                df_input["pr_id"]
                .str[:-1]
                .replace([r"C\(UniMod:4\)", r"M\(UniMod:35\)"], ["c", "m"], regex=True)
            )
        df_input["strip_seq"] = df_input["simple_seq"].str.upper()

    # seq to strip_seq
    df_pep_score = df_input[["strip_seq", "cscore_pr_" + x]].copy()
    idx_max = df_pep_score.groupby(["strip_seq"])["cscore_pr_" + x].idxmax()
    df_pep_score = df_pep_score.loc[idx_max].reset_index(drop=True)

    # row by protein group
    df = df_input[df_input["q_pr_" + x] < q_pr_cut]
    df = df[["strip_seq", "protein_group", "decoy"]]
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.merge(df_pep_score, on="strip_seq")
    df = (
        df.groupby(by=["protein_group", "decoy"])
        .agg(
            {
                ("cscore_pr_" + x): lambda g: 1 - (1 - g).prod(),
                # ('cscore_pr_' + x): lambda g: g.nlargest(1).sum(),
                "strip_seq": lambda g: list(g),
            }
        )
        .reset_index()
    )
    df = df.rename(columns={("cscore_pr_" + x): ("cscore_pg_" + x)})

    # q
    df = df.sort_values(by=("cscore_pg_" + x), ascending=False, ignore_index=True)
    target_num = (df.decoy == 0).cumsum()
    decoy_num = (df.decoy == 1).cumsum()
    target_num[target_num == 0] = 1
    df["q_pg_" + x] = decoy_num / target_num
    df["q_pg_" + x] = df["q_pg_" + x][::-1].cummin()

    df = df[["protein_group", "decoy", "cscore_pg_" + x, "q_pg_" + x]]

    # return
    df_result = df_input.merge(df, on=["protein_group", "decoy"], how="left")
    not_in_range = df_result["q_pg_" + x].isna()
    df_result.loc[not_in_range, "cscore_pg_" + x] = 0.0
    df_result.loc[not_in_range, "q_pg_" + x] = 1

    df_na["cscore_pg_" + x] = np.float32(0.0)
    df_na["q_pg_" + x] = np.float32(1.0)

    df_result = pd.concat([df_result, df_na], axis=0, ignore_index=True)

    return df_result
