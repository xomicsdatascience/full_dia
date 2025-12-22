import warnings

import numpy as np
import pandas as pd
import torch
from numba.core.errors import NumbaPerformanceWarning

from full_dia import (
    calib,
    cfg,
    cross,
    deepmap,
    fdr,
    fxic,
    library,
    polish,
    quant,
    scoring,
    tims,
    utils,
)
from full_dia.decoy import cal_fg_mz_iso, make_decoys
from full_dia.log import Logger
from full_dia.refine import refine_models
from full_dia.tims import load_ms
from full_dia.utils import init_single_ws

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=NumbaPerformanceWarning)
np.set_printoptions(suppress=True)

try:
    _ = profile
except NameError:

    def profile(func):
        return func


logger = Logger.get_logger()


@profile
def seek_seed(
    df_target: pd.DataFrame, ms: tims.Tims, model_center: torch.nn.Module
) -> pd.DataFrame:
    """
    Seek the best elution group for each pr using SA scoring methods.
    Then, model_center scores the elution group.
    Obviously, these elution groups may contain many false positives, but they can be used as seeds for calibration.

    Parameters
    ----------
    df_target : pd.DataFrame
        The identification object from library.

    ms : tims.Tims
        MS data.

    model_center : torch.nn.Module
        DeepProfile will score the coelution consistency of the elution group.

    Returns
    -------
    df : pd.DataFrame
        One precursor will have one elution group.
    """
    df_good = []
    swath_id_v = np.sort(df_target["swath_id"].unique())
    for swath_id in swath_id_v:
        df_swath = df_target[df_target["swath_id"] == swath_id]
        df_swath = df_swath.reset_index(drop=True)
        if swath_id in swath_id_v[[0, int(len(swath_id_v) / 2), -1]]:
            info = "Seek-Seed, swath_id: {}, target num: {}".format(
                swath_id, len(df_swath)
            )
            logger.info(info)

        # map_gpu
        ms1_profile, ms2_profile = ms.copy_map_to_gpu(swath_id, centroid=False)
        ms1_centroid, ms2_centroid = ms.copy_map_to_gpu(swath_id, centroid=True)
        batch_n = cfg.batch_xic_seed

        # by coelution scores using sa func
        for _, df_batch in df_swath.groupby(df_swath.index // batch_n):
            df_batch = df_batch.reset_index(drop=True)
            # [k, ions_num, n]
            locus, rts, xics = fxic.extract_xics(
                df_batch,
                ms1_centroid,
                ms2_centroid,
                cfg.tol_ppm,
                cfg.tol_im_xic,
                rt_tolerance=None,
                only_xic=True,
            )
            xics = fxic.gpu_simple_smooth(xics)
            scores_sa, scores_sa_m = fxic.cal_coelution_by_gaussion(
                xics, cfg.window_points, df_batch.fg_num.values + 2
            )
            del xics

            scores_sa, idx = torch.topk(scores_sa, 1, dim=1, sorted=False)
            idx_x = np.arange(len(locus))
            locus = locus[idx_x, idx.cpu().flatten()]
            df_batch["locus"] = locus
            df_batch["measure_rt"] = rts[idx_x, locus]
            locus_sas = scores_sa_m[idx_x, :, locus].cpu().numpy()

            # re-extract
            cycle_num = 3
            _, _, ims, mzs, _ = fxic.extract_xics(
                df_batch,
                ms1_centroid,
                ms2_centroid,
                cfg.tol_ppm,
                cfg.tol_im_xic,
                cycle_num=cycle_num,
                only_xic=False,
            )
            locus_ims = ims[:, :, int(cycle_num / 2)]
            df_batch["measure_pr_mz"] = mzs[:, 0, int(cycle_num / 2)]

            # measure_im for deep
            measure_im = fxic.cal_measure_im(locus_ims, locus_sas)
            df_batch["measure_im"] = measure_im
            scores_deep, features = deepmap.scoring_maps(
                model_center,
                df_batch,
                ms1_profile,
                ms2_profile,
                cfg.map_cycle_dim,
                cfg.map_im_gap,
                cfg.map_im_dim,
                cfg.tol_ppm,
                cfg.tol_im_map,
                neutron_num=0,
                return_feature=False,
            )
            scores_sa = scores_sa.cpu().numpy().flatten()
            scores_deep = scores_deep.cpu().numpy().flatten()
            df_batch["score_deep"] = scores_sa * scores_deep
            df_good.append(df_batch)

        utils.release_gpu_scans(ms1_profile, ms2_profile, ms1_centroid, ms2_centroid)

    df = pd.concat(df_good, axis=0, ignore_index=True)
    return df


def cal_recall_seek_seed(df_lib, ms, model_center):
    """
    For developing.
    """
    if not cfg.is_compare_mode:
        return

    # seek_seed is different with seek_locus
    df_diann = pd.read_csv(cfg.ws_single / "diann" / "report.tsv", sep="\t")
    df_diann = df_diann[df_diann["Q.Value"] < 0.01]
    df_diann["pr_id"] = df_diann["Modified.Sequence"] + df_diann[
        "Precursor.Charge"
    ].astype(str)
    df_diann = df_diann[["pr_id", "RT", "IM"]]
    if "diann_rt" not in df_lib.columns:
        df_diann["diann_rt"] = df_diann["RT"] * 60.0

    df_diann = pd.merge(df_lib, df_diann, on="pr_id")
    df_diann = df_diann.reset_index(drop=True)
    del df_diann["RT"]

    rts = ms.get_scan_rts()
    locus = np.argmin(np.abs(df_diann["diann_rt"].values[:, None] - rts), axis=1)
    df_diann["locus_diann"] = locus
    # df_diann = df_diann[df_diann.pr_id == 'VSNSGITR2'].reset_index(drop=True)

    df_good = []
    for swath_id in df_diann["swath_id"].unique():
        df_swath = df_diann[df_diann["swath_id"] == swath_id]
        df_swath = df_swath.reset_index(drop=True)

        # map_gpu
        ms1_profile, ms2_profile = ms.copy_map_to_gpu(swath_id, centroid=False)
        ms1_centroid, ms2_centroid = ms.copy_map_to_gpu(swath_id, centroid=True)

        all_rts = ms1_profile["scan_rts"]
        batch_n = cfg.batch_xic_seed

        # by coelution scores using sa func
        for _, df_batch in df_swath.groupby(df_swath.index // batch_n):
            df_batch = df_batch.reset_index(drop=True)
            ions_num = df_batch.fg_num.values + 2
            # [k, ions_num, n]
            locus, _, ims, _, xics = fxic.extract_xics(
                df_batch,
                ms1_centroid,
                ms2_centroid,
                cfg.tol_ppm,
                cfg.tol_im_xic,
                rt_tolerance=None,
            )
            # sa
            xics = fxic.gpu_simple_smooth(xics)
            sa_gausion, sa_gausion_m = fxic.cal_coelution_by_gaussion(
                xics, cfg.window_points, ions_num
            )
            xic_dp = xics.shape[-1]
            del xics

            # top-locus
            scores_sa, idx = torch.topk(sa_gausion, 1, dim=1, sorted=False)
            idx_x = np.arange(len(locus))
            locus = locus[idx_x, idx.cpu().flatten()]
            rts = all_rts[locus]

            bias = np.abs(rts - df_batch.diann_rt.values)
            df_batch["bias_coelution"] = bias

            # locus -- measure_im -- deep
            locus_ims = ims[idx_x, :, locus]
            locus_sas = sa_gausion_m[idx_x, :, locus].cpu().numpy()
            measure_ims = fxic.cal_measure_im(locus_ims, locus_sas)
            df_batch["locus"] = locus
            df_batch["measure_im"] = measure_ims
            scores_deep, features = deepmap.scoring_maps(
                model_center,
                df_batch,
                ms1_profile,
                ms2_profile,
                cfg.map_cycle_dim,
                cfg.map_im_gap,
                cfg.map_im_dim,
                cfg.tol_ppm,
                cfg.tol_im_map,
                neutron_num=0,
                return_feature=False,
            )
            scores_deep = scores_deep.cpu().numpy()

            df_batch["score_deep"] = scores_deep.max(axis=1)
            df_batch["measure_rt"] = 0
            df_good.append(df_batch)

        utils.release_gpu_scans(ms1_profile, ms2_profile, ms1_centroid, ms2_centroid)

    df = pd.concat(df_good, axis=0, ignore_index=True)
    logger.info("data points num of xics: {}".format(xic_dp))

    df1 = df[df["bias_coelution"] < cfg.locus_rt_thre]
    df1 = df1.reset_index(drop=True)
    logger.info("In seek_seed, Recall after top-1: ")
    utils.cal_acc_recall(cfg.ws_single, df1, diann_q_pr=0.01)


@profile
def seek_locus(
    df_target: pd.DataFrame,
    ms: tims.Tims,
    model_center: torch.nn.Module,
    top_sa_q: float,
    top_deep_q: float,
) -> pd.DataFrame:
    """
    Seek candidate elution groups (locus) by: 1) scree with sa 2) screen with deep.

    Parameters
    ----------
    df_target : pd.DataFrame
        Provide the precursor information.

    ms : tims.Tims
        MS data.

    model_center : torch.nn.Module
        DeepProfile-14, used to score the elution consistency of monoisotope ions.

    top_sa_q : float
        First, candidate locus should have good SA scores compared to the best elution group.

    top_deep_q : float
        Second, candidate locus should have good deep scores compared to the best elution group.

    Returns
    -------
    df : pd.DataFrame
        Each row is a candidate elution group.
    """
    df_good = []
    for swath_id in df_target["swath_id"].unique():
        df_swath = df_target[df_target["swath_id"] == swath_id]
        df_swath = df_swath.reset_index(drop=True)
        if swath_id % 5 == 1:
            info = "Seek-Locus, swath_id: {}, target num: {}".format(
                swath_id, len(df_swath)
            )
            # logger.info(info)

        # map_gpu
        ms1_profile, ms2_profile = ms.copy_map_to_gpu(swath_id, centroid=False)
        ms1_centroid, ms2_centroid = ms.copy_map_to_gpu(swath_id, centroid=True)
        all_rts = ms1_profile["scan_rts"]
        batch_n = cfg.batch_xic_locus

        # by coelution scores using sa func
        for _, df_batch in df_swath.groupby(df_swath.index // batch_n):
            df_batch = df_batch.reset_index(drop=True)
            # [k, ions_num, n]
            locus, rts, xics = fxic.extract_xics(
                df_batch,
                ms1_centroid,
                ms2_centroid,
                cfg.tol_ppm,
                cfg.tol_im_xic,
                cfg.tol_rt,
                only_xic=True,
            )
            xics = fxic.gpu_simple_smooth(xics)
            scores_sa, scores_sa_m = fxic.cal_coelution_by_gaussion(
                xics, cfg.window_points, df_batch.fg_num.values + 2
            )
            scores_sa = fxic.screen_locus_by_sa(scores_sa, top_sa_q)
            if scores_sa.max() == 0:
                continue
            locus_v, locus_num, scores_sa_v, locus_sas = fxic.concat_nonzero_locus(
                locus, scores_sa, scores_sa_m
            )

            # a row is a locus, locus_num with zeros will not be indexed
            df_batch = df_batch.loc[np.repeat(df_batch.index, locus_num)]
            df_batch = df_batch.reset_index(drop=True)
            df_batch["locus"] = locus_v
            df_batch["measure_rt"] = all_rts[locus_v]

            # re-extract
            cycle_num = 3
            _, _, ims, mzs, _ = fxic.extract_xics(
                df_batch,
                ms1_centroid,
                ms2_centroid,
                cfg.tol_ppm,
                cfg.tol_im_xic,
                cycle_num=cycle_num,
                only_xic=False,
            )
            locus_ims = ims[:, :, int(cycle_num / 2)]
            measure_ims = fxic.cal_measure_im(locus_ims, locus_sas)
            df_batch["measure_im"] = measure_ims

            # deep
            scores_deep, features = deepmap.scoring_maps(
                model_center,
                df_batch,
                ms1_profile,
                ms2_profile,
                cfg.map_cycle_dim,
                cfg.map_im_gap,
                cfg.map_im_dim,
                cfg.tol_ppm,
                cfg.tol_im_map,
                neutron_num=0,
                return_feature=False,
            )
            scores_deep = scores_deep.cpu().numpy().flatten()

            df_batch["seek_score_deep"] = scores_deep
            df_batch["seek_score_sa"] = scores_sa_v
            df_batch["seek_score_sa_x_deep"] = scores_deep * scores_sa_v

            locus_num = locus_num[locus_num > 0]
            df_batch = fxic.screen_locus_by_deep(df_batch, locus_num, top_deep_q)

            df_good.append(df_batch)

        utils.release_gpu_scans(ms1_profile, ms2_profile, ms1_centroid, ms2_centroid)

    df = pd.concat(df_good, axis=0, ignore_index=True)

    info = "Seek locus: {} prs have {} candidate locus".format(
        df.pr_id.nunique(), len(df)
    )
    logger.info(info)
    utils.cal_acc_recall(cfg.ws_single, df[df.decoy == 0], diann_q_pr=0.01)

    return df


def cal_recall_seek_locus(df_lib, ms, model, tol_rt, top_sa_cut, top_deep_cut):
    """
    For developing.
    """
    if not cfg.is_compare_mode:
        return

    df_diann = pd.read_csv(cfg.ws_single / "diann" / "report.tsv", sep="\t")
    df_diann = df_diann[df_diann["Q.Value"] < 0.01]
    df_diann["pr_id"] = df_diann["Modified.Sequence"] + df_diann[
        "Precursor.Charge"
    ].astype(str)
    df_diann = df_diann[["pr_id", "RT"]]
    if "diann_rt" not in df_lib.columns:
        df_diann["diann_rt"] = df_diann["RT"] * 60.0

    df_diann = pd.merge(df_lib, df_diann, on="pr_id")
    df_diann = df_diann.reset_index(drop=True)
    del df_diann["RT"]

    rts = ms.get_scan_rts()
    locus = np.argmin(np.abs(df_diann["diann_rt"].values[:, None] - rts), axis=1)
    df_diann["diann_locus"] = locus

    # df_diann = df_diann[df_diann.pr_id == 'ALNIETAIK3'].reset_index(drop=True)

    df_good, locus_num_v = [], []
    for swath_id in df_diann["swath_id"].unique():
        df_swath = df_diann[df_diann["swath_id"] == swath_id]
        df_swath = df_swath.reset_index(drop=True)

        # map_gpu
        ms1_profile, ms2_profile = ms.copy_map_to_gpu(swath_id, centroid=False)
        ms1_centroid, ms2_centroid = ms.copy_map_to_gpu(swath_id, centroid=True)
        all_rts = ms1_profile["scan_rts"]

        # by coelution scores using sa func
        batch_n = cfg.batch_xic_locus
        for _, df_batch in df_swath.groupby(df_swath.index // batch_n):
            df_batch = df_batch.reset_index(drop=True)

            # extract xics of 13 cycles
            df_batch["locus"] = df_batch["diann_locus"]
            locus, _, xics = fxic.extract_xics(
                df_batch,
                ms1_centroid,
                ms2_centroid,
                cfg.tol_ppm,
                cfg.tol_im_xic,
                cycle_num=13,
                only_xic=True,
            )
            xics = fxic.gpu_simple_smooth(xics)
            _, sa_m = fxic.cal_coelution_by_gaussion(
                xics, cfg.window_points, df_batch.fg_num.values + 2
            )
            xics = utils.convert_numba_to_tensor(xics)
            locus_start_v, locus_end_v = fxic.estimate_xic_boundary(xics, sa_m[:, :, 6])
            df_batch["locus_start"] = locus_start_v
            df_batch["locus_end"] = locus_end_v
            df_batch["locus_span"] = locus_end_v - locus_start_v

            # [k, ions_num, n]
            locus, _, ims, _, xics_rt = fxic.extract_xics(
                df_batch,
                ms1_centroid,
                ms2_centroid,
                cfg.tol_ppm,
                cfg.tol_im_xic,
                tol_rt,
            )
            xics_rt = fxic.gpu_simple_smooth(xics_rt)
            scores_sa, scores_sa_m = fxic.cal_coelution_by_gaussion(
                xics_rt, cfg.window_points, df_batch.fg_num.values + 2
            )
            scores_sa = fxic.screen_locus_by_sa(scores_sa, top_sa_cut)
            locus_v, locus_num, locus_sa_v, locus_sas = fxic.concat_nonzero_locus(
                locus, scores_sa, scores_sa_m
            )

            # a row is a locus, locus_num with zeros will not be indexed
            df_batch = df_batch.loc[np.repeat(df_batch.index, locus_num)]
            df_batch = df_batch.reset_index(drop=True)
            df_batch["locus"] = locus_v
            df_batch["score_sa"] = locus_sa_v
            df_batch["measure_rt"] = all_rts[locus_v]
            locus_num_v.append(locus_num)

            # re-extract
            cycle_num = 3
            _, _, ims, mzs, _ = fxic.extract_xics(
                df_batch,
                ms1_centroid,
                ms2_centroid,
                cfg.tol_ppm,
                cfg.tol_im_xic,
                cycle_num=cycle_num,
                only_xic=False,
            )
            locus_ims = ims[:, :, int(cycle_num / 2)]
            measure_ims = fxic.cal_measure_im(locus_ims, locus_sas)
            df_batch["measure_im"] = measure_ims

            scores_deep, features = deepmap.scoring_maps(
                model,
                df_batch,
                ms1_profile,
                ms2_profile,
                cfg.map_cycle_dim,
                cfg.map_im_gap,
                cfg.map_im_dim,
                cfg.tol_ppm,
                cfg.tol_im_map,
                neutron_num=0,
                return_feature=False,
            )
            scores_deep = scores_deep.cpu().numpy().flatten()
            df_batch["seek_score_deep"] = scores_deep
            df_batch["seek_score_sa"] = locus_sa_v
            df_batch["seek_score_sa_x_deep"] = locus_sa_v * scores_deep

            df_good.append(df_batch)

        utils.release_gpu_scans(ms1_profile, ms2_profile, ms1_centroid, ms2_centroid)

    df = pd.concat(df_good, axis=0, ignore_index=True)
    logger.info("data points num of xics: {}".format(xics_rt.shape[2]))

    # recall after sa
    info = "Seek-Locus-by-sa-{}: ≈{:.2f}locus/pr, recall: ".format(
        top_sa_cut, len(df) / len(df_diann)
    )
    logger.info(info)
    utils.cal_acc_recall(cfg.ws_single, df, diann_q_pr=0.01)

    # recall after sa + deep
    locus_num = np.concatenate(locus_num_v)
    locus_num = locus_num[locus_num > 0]
    assert np.all(df.groupby(by="pr_id", sort=False).size() == locus_num)
    df = fxic.screen_locus_by_deep(df, locus_num, top_deep_cut)
    info = "Seek-Locus-by-sa-{}-deep-{}: ≈{:.2f}locus/pr, recall: ".format(
        top_sa_cut, top_deep_cut, len(df) / len(df_diann)
    )
    logger.info(info)
    utils.cal_acc_recall(cfg.ws_single, df, diann_q_pr=0.01)

    df = df.drop_duplicates(subset="pr_id", ignore_index=True)
    locus_span = df["locus_span"].median()
    info = "locus_span: {}".format(int(locus_span))
    logger.info(info)


def update_tolerance(
    df_lib: pd.DataFrame,
    ms: tims.Tims,
    model_center: torch.nn.Module,
    model_big: torch.nn.Module,
    sample_ratio: float,
) -> None:
    """
    Update the tolerance based on identifications of a subset of target peptides.

    Parameters
    ----------
    df_lib : pd.DataFrame
        The raw library.

    ms : tims.Tims
        MS data.

    model_center : torch.nn.Module
        DeepProfile-14, used to score the elution consistency of monoisotopes.

    model_big : torch.nn.Module
        DeepProfile-56, used to score the elution consistency of monoisotopes + isotopes.

    sample_ratio : float
        Sample the subset of library to expedite the update.

    Returns
    -------
    None. The global tolerance values will be updated:
        cfg.tol_rt, cfg.tol_im_xic, cfg.tol_ppm
    """
    target_num = int(min(cfg.target_batch_max, sample_ratio * len(df_lib)))
    logger.info(f"Estimate the tolerance using {target_num} prs ...")

    df = df_lib.sample(n=target_num, random_state=42, replace=False)
    df = df.reset_index(drop=True)

    # seek for targets and decoys
    df1 = make_decoys(df, cfg.fg_num, method="mutate")
    df = pd.concat([df, df1]).reset_index(drop=True)
    df = seek_locus(df, ms, model_center, cfg.top_sa_cut, cfg.top_deep_cut)
    df = cal_fg_mz_iso(df)
    df = scoring.score_locus(df, ms, model_center, model_big)
    df, model_nn, scaler = fdr.cal_q_pr_batch(df, 50, 12)
    df = df[df["decoy"] == 0]
    for q_cut in [0.01, 0.02, 0.03, 0.04, 0.05]:
        ids = sum(df["q_pr_run"] < q_cut)
        if ids >= 50:
            break
    df = df[df["q_pr_run"] < q_cut].reset_index(drop=True)
    logger.info(f"Estimate the tolerance using {len(df)} prs at {q_cut:.2f} FDR:")

    if len(df) > 50:
        # estimate tol
        x = df["score_rt_abs"].values
        cut1 = x.mean() + 3.0 * x.std()
        cut2 = np.percentile(x, 95)
        tol_rt = max(cut1, cut2)

        cycle_time = ms.get_cycle_time()
        cycle_num = int(tol_rt * 2 / cycle_time)
        if cycle_num < 23:
            tol_rt = 23 * cycle_time / 2

        x = df["score_imbias_0"].values
        x = x[(x > x.min()) & (x < x.max())]
        cut1 = x.mean() + 3.0 * x.std()
        cut2 = np.percentile(x, 95)
        tol_im = max(cut1, cut2)

        x = df["score_ppm_0"].values
        x = x[(x > x.min()) & (x < x.max())]
        cut1 = x.mean() + 3.0 * x.std()
        cut2 = np.percentile(x, 95)
        tol_ppm = max(cut1, cut2)

        cfg.tol_rt = tol_rt
        cfg.tol_im_xic = tol_im
        cfg.tol_ppm = tol_ppm
    else:
        cfg.tol_im_xic = cfg.tol_im_xic_after_calib

    info = "tol_rt: {:.2f}, tol_im: {:.3f}, tol_ppm: {:.2f}".format(
        cfg.tol_rt, cfg.tol_im_xic, cfg.tol_ppm
    )
    logger.info(info)


def select_required_and_all_targets(df: pd.DataFrame) -> tuple:
    """
    Select good target and decoy peps for FDR calculation.
    Select all target to save, which avoids the second extraction in global analysis.

    Parameters
    ----------
    df : pd.DataFrame
        The identification result for one batch.

    Returns
    -------
    tuple
        df_main : pd.DataFrame
            Good target and decoy peps.
        df_other : pd.DataFrame
            All target peps.
    """
    df_main = df[df["q_pr_run"] < cfg.rubbish_q_cut].copy()

    cols_base = [
        "pr_id",
        "pr_charge",
        "pr_index",
        "swath_id",
        "decoy",
        "locus",
        "measure_rt",
        "measure_im",
    ]
    cols_base += ["pr_mz", "score_elute_span_left", "score_elute_span_right"]
    cols_base += ["fg_mz_" + str(i) for i in range(cfg.fg_num)]
    df_other = df.loc[df["decoy"] == 0, cols_base].copy()

    return df_main, df_other


def search_core(lib: library.Library) -> None:
    """
    Search on run level:
        1. Seek seeds for calibration.
        2. Seek candidate elution groups (locus) for each precursor.
        3. Score the elution groups.
        4. Calculate the FDR on run level.
        5. Save all target precursor results and high-quality decoy precursor results.
    """
    for ws_i, ws_single in enumerate(cfg.multi_ws):
        # ws_single
        init_single_ws(ws_i, cfg.file_num, ws_single)
        run_finished = (cfg.dir_out_global / (ws_single.name + ".parquet")).exists()
        if run_finished and not cfg.is_overwrite:
            continue

        ms = load_ms(ws_single)
        # utils.get_diann_info(ws_single)

        # set tol_rt
        cfg.tol_rt = ms.get_scan_rts()[-1] * cfg.tol_rt_ratio
        cycle_time = np.diff(ms.get_scan_rts()).mean()
        cfg.locus_rt_thre = cycle_time * cfg.locus_valid_num

        # polish lib
        df_lib = lib.polish_lib_by_swath(
            ms.get_dia_quadrupole(),
            #     ws_diann= cfg.ws_single  # for debug
        )

        # load the pretrained models
        model_center, model_big = deepmap.load_models()

        # seek seed throughout the LC time
        cal_recall_seek_seed(df_lib, ms, model_center)
        df_seed = seek_seed(df_lib, ms, model_center)
        utils.save_as_pkl(df_seed, "df_seed.pkl")
        # df_seed = pd.read_pickle(cfg.dir_out_single / 'df_seed.pkl')

        # update tol
        df_seed, df_lib = calib.calib_rt(df_seed, df_lib)
        df_seed, df_lib = calib.calib_im(df_seed, df_lib)
        df_seed = calib.calib_mz(df_seed, ms)
        del df_seed
        update_tolerance(df_lib, ms, model_center, model_big, cfg.sample_ratio)

        # check params for seek_locus
        cal_recall_seek_locus(
            df_lib,
            ms,
            model_center,
            tol_rt=cfg.tol_rt,
            top_sa_cut=cfg.top_sa_cut,
            top_deep_cut=cfg.top_deep_cut,
        )

        # batch division
        batch_num = int(np.ceil(len(df_lib) / cfg.target_batch_max))
        rows_idx = np.array_split(df_lib.index.values, batch_num)
        df_main_v, df_other_v = [], []
        for batch_idx in range(len(rows_idx)):
            if batch_idx % 3 == 0:
                logger.disabled = False
                info = "-------------Run-{}/{}-Batch-{}/{}-------------".format(
                    ws_i + 1, cfg.file_num, batch_idx + 1, batch_num
                )
                logger.info(info)
            else:
                logger.disabled = True

            df = df_lib.iloc[rows_idx[batch_idx]].reset_index(drop=True)

            # seek for targets and decoys
            df1 = make_decoys(df, cfg.fg_num, method="mutate")
            df = pd.concat([df, df1]).reset_index(drop=True)
            df = seek_locus(df, ms, model_center, cfg.top_sa_cut, cfg.top_deep_cut)

            # scoring by first round
            df = cal_fg_mz_iso(df)
            df = scoring.score_locus(df, ms, model_center, model_big)
            if batch_idx == 0:
                df, model_nn, scaler = fdr.cal_q_pr_batch(df, 50, 12)
                fdr.adjust_rubbish_q(df, batch_num)
            else:
                df, _, _ = fdr.cal_q_pr_batch(df, 50, 12, model_nn, scaler)
            df_main, df_other = select_required_and_all_targets(df)
            utils.print_ids(df_main, cfg.rubbish_q_cut, "pr", "run")
            df_main_v.append(df_main)
            df_other_v.append(df_other)
        df_main = pd.concat(df_main_v, axis=0, ignore_index=True)
        df_other = pd.concat(df_other_v, axis=0, ignore_index=True)
        logger.disabled = False

        logger.info("--------------------Concat batches:---------------------")
        df_main = cross.drop_batches_mismatch(df_main)
        df_main = fdr.cal_q_pr_combined(df_main, 50, 12)
        utils.print_ids(df_main, 1, "pr", "run")
        logger.info("--------------------------------------------------------")
        utils.save_as_pkl(df_main, "df_scores1.pkl")
        # df = pd.read_pickle(cfg.dir_out_single / 'df_scores1.pkl')

        # retrain models
        model_center, model_big, model_mall = refine_models(
            df_main, ms, model_center, model_big
        )

        # scoring by second round
        df_main = scoring.update_scores(
            df_main, ms, model_center, model_big, model_mall
        )
        utils.save_as_pkl(df_main, "df_scores2.pkl")
        # df_main = pd.read_pickle(cfg.dir_out_single / 'df_scores2.pkl')

        # scoring using DIA-NN sa and quant fg ions
        df_main = df_main[df_main["score_deep_center_sub_left_refine"] > 0].reset_index(
            drop=True
        )
        logger.info(
            "Calculating coelution with the best-profile and quantifying ions for all prs..."
        )
        df_main = quant.quant_center_ions(df_main, ms)
        df_other = quant.quant_center_ions(df_other, ms)
        utils.save_as_pkl(df_main, "df_main.pkl")
        utils.save_as_pkl(df_other, "df_other.pkl")
        # df_main = pd.read_pickle(cfg.dir_out_single / 'df_main.pkl')
        # df_other = pd.read_pickle(cfg.dir_out_single / 'df_other.pkl')

        # update FDR-Pr for second round
        df_main = fdr.cal_q_pr_combined(df_main, 50, 12)
        utils.print_ids(df_main, 1, "pr", "run")
        utils.save_as_pkl(df_main, "df_fdr1.pkl")
        # df_main = pd.read_pickle(cfg.dir_out_single / 'df_fdr1.pkl')

        # polish target prs and update FDR again
        df_main = polish.polish_prs(df_main)
        df_main = fdr.cal_q_pr_combined(df_main, 50, 12)
        df_main = polish.polish_prs(df_main)
        utils.print_ids(df_main, 1, "pr", "run")
        utils.save_as_pkl(df_main, "df_fdr2.pkl")
        # df_main = pd.read_pickle(cfg.dir_out_single / 'df_fdr2.pkl')

        # save result
        logger.info("Saving run-specific result as parquet...")
        utils.clean_and_save(df_main, df_other, ws_single)
        logger.info("Saving finished.")

        # release within loop
        del df_lib, df, ms
        cfg.tol_im_xic = cfg.tol_im_xic_before_calib
        cfg.tol_ppm = cfg.tol_ppm_before_calib
