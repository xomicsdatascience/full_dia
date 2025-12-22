import numpy as np
import pandas as pd
import torch
from numba import cuda, jit

from full_dia import cfg, deepmall, deepmap, fxic, tims, utils
from full_dia.log import Logger

try:
    _ = profile
except NameError:

    def profile(func):
        return func


logger = Logger.get_logger()


@profile
def score_locus(
    df_target: pd.DataFrame,
    ms: tims.Tims,
    model_center: torch.nn.Module,
    model_big: torch.nn.Module,
) -> pd.DataFrame:
    """
    Calculate function-based and learning-based scores for PSMs.

    Parameters
    ----------
    df_target : pd.DataFrame
        Provide the PSM information.

    ms : tims.Tims
        MS data.

    model_center : torch.nn.Module
        DeepProfile-14

    model_big : torch.nn.Module
        DeepProfile-56

    Returns
    -------
    df : pd.DataFrame
        Scores have been appended to the DataFrame in columns prefixed with "score_".
    """
    df_good = []
    for swath_id in df_target["swath_id"].unique():
        df_swath = df_target[df_target["swath_id"] == swath_id]
        df_swath = df_swath.reset_index(drop=True)

        # map_gpu
        ms1_profile, ms2_profile = ms.copy_map_to_gpu(swath_id, centroid=False)
        ms1_centroid, ms2_centroid = ms.copy_map_to_gpu(swath_id, centroid=True)

        batch_n = cfg.batch_deep_big

        # may split two locus that belong to a pr
        for _, df_batch in df_swath.groupby(df_swath.index // batch_n):
            df_batch = df_batch.reset_index(drop=True)
            # deep scores and deep features
            scores_deep_v, features_deep_v = deepmap.extract_scoring_big(
                model_center,
                model_big,
                df_batch,
                ms1_profile,
                ms2_profile,
                cfg.map_cycle_dim,
                cfg.map_im_gap,
                cfg.map_im_dim,
                cfg.tol_ppm,
                cfg.tol_im_map,
            )
            _, rts, ims_v, mzs_v, xics_v = fxic.extract_xics(
                df_batch,
                ms1_centroid,
                ms2_centroid,
                cfg.tol_ppm,
                cfg.tol_im_xic,
                cycle_num=13,
                scope="big",
            )
            _, _, xics_ppm1 = fxic.extract_xics(
                df_batch,
                ms1_centroid,
                ms2_centroid,
                cfg.tol_ppm * 0.5,
                cfg.tol_im_xic,
                cycle_num=13,
                only_xic=True,
            )
            _, _, xics_ppm2 = fxic.extract_xics(
                df_batch,
                ms1_centroid,
                ms2_centroid,
                cfg.tol_ppm * 0.25,
                cfg.tol_im_xic,
                cycle_num=13,
                only_xic=True,
            )
            # sa scores
            df_batch = scoring_other_elution(df_batch, xics_v[0], x="left")
            df_batch, xics = scoring_main_elution(df_batch, xics_v[1], x="center")
            df_batch = scoring_other_elution(df_batch, xics_v[2], x="1H")
            df_batch = scoring_other_elution(df_batch, xics_v[3], x="2H")

            df_batch, _ = scoring_main_elution(df_batch, xics_ppm1, x="center_p1")
            df_batch, _ = scoring_main_elution(df_batch, xics_ppm2, x="center_p2")

            # intensity, similarity, height ratio, area, snr
            df_batch = scoring_center_snr(df_batch, xics)
            df_batch = scoring_xic_intensity(df_batch, xics, rts)

            # deep
            df_batch = scoring_by_deep_prob(df_batch, scores_deep_v, x="pre")
            df_batch = scoring_by_deep_layer(df_batch, features_deep_v, x="pre")
            # rt
            df_batch = scoring_rt(df_batch)
            # im
            df_batch = scoring_center_im(df_batch, ims_v[1])
            # mz
            df_batch = scoring_center_mz(df_batch, mzs_v[1])
            # cross scores
            df_batch = scoring_by_cross(df_batch)

            df_good.append(df_batch)

        utils.release_gpu_scans(ms1_profile, ms2_profile, ms1_centroid, ms2_centroid)

    df = pd.concat(df_good, axis=0, ignore_index=True)
    df = scoring_putatives(df)  # competitive for two locus from a pr
    df = scoring_meta(df)  # meta scores
    return df


def scoring_by_deep_prob(
    df_batch: pd.DataFrame, scores_deep_v: list, x: str
) -> pd.DataFrame:
    """
    Append the inference scores of DeepProfile to df.

    Parameters
    ----------
    df_batch : pd.DataFrame
        The object.

    scores_deep_v : list
        The inference scores of DeepProfile.

    x : str
        "pre": scores are from the pretrain models.
        "refine": scores are from the refinement models.
        "refine_p1": scores are from the refinement models with 0.5 * ppm.
        "refine_p2": scores are from the refinement models with 0.25 * ppm.

    Returns
    -------
    df : pd.DataFrame
        The inference scores of DeepProfile have been appended.
    """
    if scores_deep_v[0] is not None:
        df_batch[f"score_left_deep_{x}"] = scores_deep_v[0]
    if scores_deep_v[1] is not None:
        df_batch[f"score_center_deep_{x}"] = scores_deep_v[1]
    if scores_deep_v[2] is not None:
        df_batch[f"score_1H_deep_{x}"] = scores_deep_v[2]
    if scores_deep_v[3] is not None:
        df_batch[f"score_2H_deep_{x}"] = scores_deep_v[3]
    if scores_deep_v[4] is not None:
        df_batch[f"score_big_deep_{x}"] = scores_deep_v[4]

    return df_batch


@profile
def scoring_by_deep_layer(
    df_batch: pd.DataFrame, features_deep_v: list, x: str
) -> pd.DataFrame:
    """
    Append the feature layers scores of DeepProfile to df.

    Parameters
    ----------
    df_batch : pd.DataFrame
        The object.

    features_deep_v : list
        The feature layers scores of DeepProfile.

    x : str
        "pre": scores are from the pretrain models.
        "refine_p1": scores are from the refinement models with 0.5 * ppm.
        "refine_p2": scores are from the refinement models with 0.25 * ppm.

    Returns
    -------
    df : pd.DataFrame
        The feature layers scores of DeepProfile have been appended.
    """
    owned = 0
    for features in features_deep_v:
        m = features.shape[-1]
        columns = [f"score_ft_deep_{x}_{i}" for i in range(owned, owned + m)]
        df_batch[columns] = features
        owned += m

    return df_batch


@profile
def scoring_other_elution(
    df_batch: pd.DataFrame, xics: torch.Tensor, x: str
) -> pd.DataFrame:
    """
    Calculate the following elution scores based on the isotope types specified by x:
        x: ['left', '1H', '2H']
        1. sa for each of the 14 ions
        2. mean value of 14 ions
        3. mean value of top-6
        4. mean value w/o norm of remaining ions
    """
    if xics is None:
        return df_batch

    fg_num = df_batch["fg_num"].values

    xics = cuda.as_cuda_array(xics)
    xics = fxic.gpu_simple_smooth(xics)
    coelutions, elutions = fxic.cal_coelution_by_gaussion(
        xics, cfg.window_points, 2 + fg_num
    )

    center_idx = int(xics.shape[-1] / 2)
    idx_x = np.arange(len(df_batch))

    coelutions = coelutions[idx_x, center_idx].cpu().numpy()
    elutions = elutions[idx_x, :, center_idx].cpu().numpy()

    # sa for 14 ions
    m = elutions.shape[-1]
    columns = [f"score_{x}_elution_{i}" for i in range(m)]
    df_batch[columns] = elutions

    # mean of 14 ions
    df_batch[f"score_{x}_coelution"] = coelutions

    # mean of top-6 ions
    fg_elutions = elutions[:, 2:].copy()
    fg_elutions_6 = fg_elutions[:, :6].copy()
    df_batch[f"score_{x}_coelution_top6"] = fg_elutions_6.sum(axis=1)

    # mean w/o norm for remaining ions
    elution_rest = fg_elutions[:, 6:].sum(axis=1)
    elution_rest_norm = elution_rest / (fg_num - 6 - 1e-7)
    elution_rest_norm[elution_rest_norm < 0] = 0
    elution_rest = elution_rest.astype(np.float32)
    elution_rest_norm = elution_rest_norm.astype(np.float32)
    df_batch[f"score_{x}_coelution_rest"] = elution_rest
    df_batch[f"score_{x}_coelution_rest_norm"] = elution_rest_norm

    return df_batch


@profile
def scoring_main_elution(
    df_batch: pd.DataFrame, xics: torch.Tensor, x: str
) -> pd.DataFrame:
    """
    Calculate the following elution scores based on the monoisotope types specified by x:
        x: ['center', 'center_p1', 'center_p2']
        1. The sa for each of the 14 ions
        2. mean value of 14 ions
        3. mean value of top-6
        4. mean value w/o norm of remaining ions
        5. sum of top1/2/3 b ions
    """
    fg_num = df_batch["fg_num"].values

    xics = cuda.as_cuda_array(xics)
    xics = fxic.gpu_simple_smooth(xics)
    coelutions, elutions = fxic.cal_coelution_by_gaussion(
        xics, cfg.window_points, 2 + fg_num
    )

    center_idx = int(xics.shape[-1] / 2)
    idx_x = np.arange(len(df_batch))

    coelutions = coelutions[idx_x, center_idx].cpu().numpy()
    elutions = elutions[idx_x, :, center_idx].cpu().numpy()

    # sa for 14 ions and its mean
    df_batch[f"score_{x}_coelution"] = coelutions.astype(np.float32)

    m = elutions.shape[-1]
    columns = [f"score_{x}_elution_{i}" for i in range(m)]
    df_batch[columns] = elutions

    # mean of top-6 ions; mean w/o norm for remaining ions
    fg_elutions = elutions[:, 2:].copy()
    fg_elutions_6 = fg_elutions[:, :6].copy()
    df_batch[f"score_{x}_coelution_top6"] = fg_elutions_6.sum(axis=1)

    elution_rest = fg_elutions[:, 6:].sum(axis=1)
    elution_rest_norm = elution_rest / (fg_num - 6 - 1e-7)
    elution_rest_norm[elution_rest_norm < 0] = 0
    elution_rest = elution_rest.astype(np.float32)
    elution_rest_norm = elution_rest_norm.astype(np.float32)
    df_batch[f"score_{x}_coelution_rest"] = elution_rest
    df_batch[f"score_{x}_coelution_rest_norm"] = elution_rest_norm

    # sum of top1/2/3 b ions
    if x.find("p") == -1:  # ppm-10/5 are not available
        cols_anno = ["fg_anno_" + str(i) for i in range(cfg.fg_num)]
        fg_anno = df_batch[cols_anno].values
        fg_type = fg_anno // 1000
        fg_elutions[fg_type != 1] = 0  # non-b series set to 0
        fg_elutions = np.sort(fg_elutions, axis=1)[:, ::-1]
        df_batch[f"score_{x}_elution_b_top1"] = fg_elutions[:, 0]
        df_batch[f"score_{x}_elution_b_top2"] = fg_elutions[:, :2].sum(axis=1)
        df_batch[f"score_{x}_elution_b_top3"] = fg_elutions[:, :3].sum(axis=1)

    return df_batch, utils.convert_numba_to_tensor(xics)


@profile
def scoring_xic_intensity(
    df_batch: pd.DataFrame, xics: torch.Tensor, rts: np.ndarray
) -> pd.DataFrame:
    """
    Calculate the intensity related scores.
    Only top-6 intensities are consideration.
    apex intensities: ms2_relative, ms2_total, ms1/ms2, similarity
    profile areas: ms2_relative, ms2_total, ms1/ms2, similarity
    """
    center_idx = int(xics.shape[-1] / 2)
    cols = ["score_center_elution_" + str(i) for i in range(14)]
    elutions = df_batch[cols].values + 1e-7

    # boundary
    sa_m = torch.from_numpy(elutions).to(cfg.gpu_id)
    locus_start_v, locus_end_v = fxic.estimate_xic_boundary(xics, sa_m)
    locus_start_v = locus_start_v.astype(np.int8)
    locus_end_v = locus_end_v.astype(np.int8)
    df_batch["score_elute_span_left"] = locus_start_v
    df_batch["score_elute_span_right"] = locus_end_v
    df_batch["score_elute_span"] = locus_end_v - locus_start_v

    # outside of boundary set to 0
    xics = xics[:, :8, :].cpu().numpy()
    mask1 = np.arange(xics.shape[2]) >= locus_start_v[:, None, None]
    mask2 = np.arange(xics.shape[2]) <= locus_end_v[:, None, None]
    xics = xics * (mask1 & mask2)

    # intensity：ms1 and ms2_total
    ms1_heights = xics[:, 0, center_idx]
    unfrag_heights = xics[:, 1, center_idx]
    ms2_heights = xics[:, 2:, center_idx]
    ms2_height_sum = ms2_heights.sum(axis=1)
    df_batch["score_intensity_ms1"] = np.log(ms1_heights + 1.0)
    df_batch["score_intensity_unfrag"] = np.log(unfrag_heights + 1.0)
    df_batch["score_intensity_ms2_total"] = np.log(ms2_height_sum + 1.0)

    # intensity: ms2_relative
    row_max = np.max(ms2_heights, axis=1, keepdims=True) + 1e-7
    ms2_heights_norm = ms2_heights / row_max
    m = ms2_heights_norm.shape[-1]
    columns = ["score_intensity_ms2_relative_" + str(i) for i in range(m)]
    df_batch[columns] = ms2_heights_norm

    # intensity: ms1/ms2
    ms1_ms2_ratio = ms1_heights / (ms2_height_sum + 1e-7)
    df_batch["score_intensity_ms1_ms2_ratio"] = np.log(ms1_ms2_ratio + 1e-7)

    # intensity: similarity
    cols_height = ["fg_height_" + str(i) for i in range(6)]
    ms2_lib = df_batch[cols_height].values
    pcc = utils.cal_sa_by_np(ms2_lib, ms2_heights_norm)
    df_batch["score_intensity_similarity"] = pcc
    df_batch["score_intensity_similarity_cube"] = pcc**3

    # area
    rts = np.repeat(rts[:, np.newaxis, :], xics.shape[1], axis=1)
    areas = np.trapz(xics, x=rts, axis=2)

    # area: ms1, unfrag, ms2
    ms1_heights = areas[:, 0]
    unfrag_heights = areas[:, 1]
    ms2_heights = areas[:, 2:]
    ms2_height_sum = ms2_heights.sum(axis=1)
    df_batch["score_area_ms1"] = np.log(ms1_heights + 1.0)
    df_batch["score_area_unfrag"] = np.log(unfrag_heights + 1.0)
    df_batch["score_area_ms2_total"] = np.log(ms2_height_sum + 1.0)

    # area: ms2_relative
    row_max = np.max(ms2_heights, axis=1, keepdims=True) + 1e-7
    ms2_heights_norm = ms2_heights / row_max
    m = ms2_heights_norm.shape[-1]
    columns = ["score_area_relative_" + str(i) for i in range(m)]
    df_batch[columns] = ms2_heights_norm

    # area: ms1/ms2
    ms1_ms2_ratio = ms1_heights / (ms2_height_sum + 1e-7)
    df_batch["score_area_ms1_ms2_ratio"] = np.log(ms1_ms2_ratio + 1e-7)

    # area: similarity
    pcc = utils.cal_sa_by_np(ms2_lib, ms2_heights_norm)
    df_batch["score_area_similarity"] = pcc
    df_batch["score_area_similarity_cube"] = pcc**3

    return df_batch


def scoring_rt(df_batch: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate RT related scores.
    """
    measure_rts = df_batch["measure_rt"].values
    pred_rts = df_batch["pred_rt"].values
    rt_bias = np.abs(pred_rts - measure_rts)

    df_batch["score_measure_rt"] = measure_rts
    df_batch["score_pred_rt"] = pred_rts
    df_batch["score_rt_abs"] = rt_bias
    df_batch["score_rt_power"] = rt_bias**2
    df_batch["score_rt_root"] = rt_bias**0.5
    df_batch["score_rt_log"] = np.log(rt_bias + 1.0)
    small = np.minimum(measure_rts, pred_rts)
    big = np.maximum(measure_rts, pred_rts)
    df_batch["score_rt_ratio"] = small / big

    return df_batch


@profile
def scoring_center_snr(df_batch: pd.DataFrame, xics: torch.Tensor) -> pd.DataFrame:
    """
    Calculate SNR related scores with the center cycle MS/MS.
    Signal is the apex intensiy, noise is the median of profile.
        1. snrs for 14 ions
        2. mean
        3. mean weighting by sa
        4. mean of top-6 weighting by sa
    """
    center_idx = int(xics.shape[-1] / 2)
    signals = xics[:, :, center_idx - 1 : center_idx + 2].amax(dim=2)
    noises = xics.median(dim=2)[0]
    snr = (signals + 1) / (noises + 1)
    snr = snr.cpu().numpy()
    fg_num = df_batch["fg_num"].values

    # 1. snrs for 14 ions
    m = snr.shape[-1]
    columns = ["score_center_snr_" + str(i) for i in range(m)]
    df_batch[columns] = np.log(snr)

    # 2. mean
    snr_average = snr.sum(axis=1) / (2 + fg_num)
    df_batch["score_center_snr_average1"] = np.log(snr_average + 1e-7)

    # 3. mean weighting by sa
    cols = ["score_center_elution_" + str(i) for i in range(14)]
    elutions = df_batch[cols].values + 1e-7
    snr_average = np.average(snr, weights=elutions, axis=1)
    df_batch["score_center_snr_average2"] = np.log(snr_average + 1e-7)

    # 4. mean of top-6 weighting by sa
    snr_fg = snr[:, 2:8]
    fg_elutions_6 = elutions[:, 2:8]
    snr_average = np.average(snr_fg, weights=fg_elutions_6, axis=1)
    df_batch["score_center_snr_average3"] = np.log(snr_average + 1e-7)

    return df_batch


@profile
def scoring_center_im(df_batch: pd.DataFrame, ims_input: np.ndarray) -> pd.DataFrame:
    """
    Calculate mobility related scores with the center cycle MS/MS.
        1. imbias for 14 ions
        2. mean
        3. mean weighting by sa
        4. mean of top-6 weighting by sa
    """
    center_idx = int(ims_input.shape[-1] / 2)
    ims = ims_input[:, :, center_idx]
    ims[ims < 0.0] = 0.0  # missing value -- 0
    fg_num = df_batch["fg_num"].values

    # im for precursor
    df_batch["score_pred_im"] = df_batch["pred_im"]
    df_batch["score_measure_im"] = df_batch["measure_im"]

    # imbias for ions，missing value -- tol
    bias = np.abs(ims - df_batch["pred_im"].values[:, None])
    bias[bias > cfg.tol_im_xic] = cfg.tol_im_xic

    # 1. imbias for 14 ions
    m = bias.shape[-1]
    columns = ["score_imbias_" + str(i) for i in range(m)]
    df_batch[columns] = bias

    # 2. mean
    bias_ms2 = bias[:, 2:]
    bias_ms2[fg_num[:, None] <= np.arange(bias_ms2.shape[1])] = 0
    bias_average = bias_ms2.sum(axis=1) / fg_num
    df_batch["score_imbias_average1"] = bias_average

    # 3. mean weighting by sa
    cols = ["score_center_elution_" + str(i) for i in range(14)]
    elutions = df_batch[cols].values + 1e-7
    bias_average = np.average(bias_ms2, weights=elutions[:, 2:], axis=1)
    df_batch["score_imbias_average2"] = bias_average

    # 4. mean of top-6 weighting by sa
    fg_elutions_6 = elutions[:, 2:8]
    bias_ms2 = bias_ms2[:, :6]
    bias_average = np.average(bias_ms2, weights=fg_elutions_6, axis=1)
    df_batch["score_imbias_average3"] = bias_average

    return df_batch


@profile
def scoring_center_mz(df_batch: pd.DataFrame, mzs_input: np.ndarray) -> pd.DataFrame:
    """
    Calculate ppm related scores with the center cycle MS/MS.
        1. ppm for 14 ions
        2. mean
        3. mean weighting by sa
        4. mean of top-6 weighting by sa
    """
    center_idx = int(mzs_input.shape[-1] / 2)
    mzs = mzs_input[:, :, center_idx]
    mzs[mzs < 0.0] = 0.0  # missing value -- 0
    fg_num = df_batch["fg_num"].values

    # mz for precursor
    df_batch["score_pr_mz"] = df_batch["pr_mz"]
    df_batch["score_pr_mz_measure"] = mzs[:, 0]

    # ppm
    mzs_pr = df_batch["pr_mz"].values.reshape(-1, 1)
    cols_center = ["fg_mz_" + str(i) for i in range(cfg.fg_num)]
    mzs_fg = df_batch[cols_center].values
    mzs_pred = np.concatenate([mzs_pr, mzs_pr, mzs_fg], axis=1)
    ppm = 1e6 * np.abs(mzs_pred - mzs) / (mzs_pred + 1e-7)
    ppm[ppm > cfg.tol_ppm] = cfg.tol_ppm

    # 1. ppm for 14 ions
    m = ppm.shape[-1]
    columns = ["score_ppm_" + str(i) for i in range(m)]
    df_batch[columns] = ppm

    # 2. mean
    ppm_ms2 = ppm[:, 2:]
    ppm_ms2[fg_num[:, None] <= np.arange(ppm_ms2.shape[1])] = 0
    ppm_average = ppm_ms2.sum(axis=1) / fg_num
    df_batch["score_ppm_average1"] = ppm_average

    # 3. mean weighting by sa
    cols = ["score_center_elution_" + str(i) for i in range(14)]
    elutions = df_batch[cols].values + 1e-7
    ppm_average = np.average(ppm_ms2, weights=elutions[:, 2:], axis=1)
    df_batch["score_ppm_average2"] = ppm_average

    # 4. mean of top-6 weighting by sa
    fg_elutions_6 = elutions[:, 2:8]
    ppm_ms2 = ppm_ms2[:, :6]
    ppm_average = np.average(ppm_ms2, weights=fg_elutions_6, axis=1)
    df_batch["score_ppm_average3"] = ppm_average

    return df_batch


@profile
def scoring_meta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate peptide meta information related scores:
        1. mz (scoring_center_mz)
        2. charge(one-hot encoding using 1, 2, 3, 4)
        3. sequence length
        4. fg_num
        5. library fragment ions intensities
    """
    c = pd.Categorical(df["pr_charge"], categories=[1, 2, 3, 4])
    pr_charges = pd.get_dummies(c, prefix="score_pr_charge").astype(np.int8)
    df = pd.concat([df, pr_charges], axis=1)

    df["score_pr_len"] = df["simple_seq"].str.len().astype(np.int8)
    df["score_fg_num"] = df["fg_num"].astype(np.int8)

    # frag info：height
    cols_height = ["fg_height_" + str(i) for i in range(1, cfg.fg_num)]
    height = df[cols_height].values  # [k, m]
    columns = ["score_lib_height_" + str(i) for i in range(height.shape[-1])]
    df[columns] = height

    return df


@jit(nopython=True, nogil=True)
def numba_scoring_putatives(groups, sa_v, center_v, big_v):
    """
    Use Numba to accelerate the computation of the maximum and sum scores across different candidate peak groups for the same precursor.
    """
    result_sa_max = np.empty(len(groups), dtype=sa_v.dtype)
    result_sa_sum = np.empty(len(groups), dtype=sa_v.dtype)
    result_center_max = np.empty(len(groups), dtype=sa_v.dtype)
    result_center_sum = np.empty(len(groups), dtype=sa_v.dtype)
    result_big_max = np.empty(len(groups), dtype=sa_v.dtype)
    result_big_sum = np.empty(len(groups), dtype=sa_v.dtype)

    current_group = groups[0]
    sa_max = sa_v[0]
    sa_sum = sa_v[0]
    center_max = center_v[0]
    center_sum = center_v[0]
    big_max = big_v[0]
    big_sum = big_v[0]

    start_idx = 0

    for i in range(1, len(groups)):
        if groups[i] != current_group:
            for j in range(start_idx, i):
                result_sa_max[j] = sa_max
                result_center_max[j] = center_max
                result_big_max[j] = big_max
                result_sa_sum[j] = sa_sum
                result_center_sum[j] = center_sum
                result_big_sum[j] = big_sum

            current_group = groups[i]
            sa_max = sa_v[i]
            sa_sum = sa_v[i]
            center_max = center_v[i]
            center_sum = center_v[i]
            big_max = big_v[i]
            big_sum = big_v[i]
            start_idx = i
        else:
            sa_max = max(sa_max, sa_v[i])
            center_max = max(center_max, center_v[i])
            big_max = max(big_max, big_v[i])

            sa_sum += sa_v[i]
            center_sum += center_v[i]
            big_sum += big_v[i]

    for j in range(start_idx, len(groups)):
        result_sa_max[j] = sa_max
        result_center_max[j] = center_max
        result_big_max[j] = big_max
        result_sa_sum[j] = sa_sum
        result_center_sum[j] = center_sum
        result_big_sum[j] = big_sum

    return (
        result_sa_max,
        result_center_max,
        result_big_max,
        result_sa_sum,
        result_center_sum,
        result_big_sum,
    )


@profile
def scoring_putatives(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate competition-related scores as a pr has multiple candidate elution groups:
        1. score-i - score-max
        2. np.log(score-i/score.sum)
    """
    a = 1e-7

    pr_index_v = df["pr_index"].values
    sa_v = df["score_center_coelution"].values
    center_v = df["score_center_deep_pre"].values
    big_v = df["score_big_deep_pre"].values
    (sa_max_v, center_max_v, big_max_v, sa_sum_v, center_sum_v, big_sum_v) = (
        numba_scoring_putatives(pr_index_v, sa_v, center_v, big_v)
    )
    df["score_center_coelution_putative1"] = sa_v - sa_max_v
    df["score_center_coelution_putative2"] = np.log(sa_v + a) / (sa_sum_v + a)

    df["score_center_deep_pre_putative1"] = center_v - center_max_v
    df["score_center_deep_pre_putative2"] = np.log(center_v + a) / (center_sum_v + a)

    df["score_big_deep_pre_putative1"] = big_v - big_max_v
    df["score_big_deep_pre_putative2"] = np.log(big_v + a) / (big_sum_v + a)

    # rank
    group_size = df.groupby("pr_id", sort=False).size()
    group_size_cumsum = np.concatenate([[0], np.cumsum(group_size)])
    group_rank = utils.cal_group_rank(
        df["score_big_deep_pre"].values, group_size_cumsum
    )
    df["group_rank"] = group_rank

    return df


def scoring_by_cross(df_batch: pd.DataFrame, is_update: bool = False) -> pd.DataFrame:
    """
    Compute scores combinations as additional scores:
        Before refine phase (is_update: False):
            1. sa_center - sa_left
            2. deep_center - deep_left
            3. sa_center * deep_center
            4. sa_center * deep_big
        After refine phase (is_update: True):
            1. deep_center - deep_left
            2. sa_center * deep_center
            3. sa_center * deep_big
    """
    # feature augmentation
    if not is_update:
        # raw model + non-ppm
        sa_center = df_batch["score_center_coelution"].values
        sa_left = df_batch["score_left_coelution"].values
        deep_center = df_batch["score_center_deep_pre"].values
        deep_left = df_batch["score_left_deep_pre"].values
        deep_big = df_batch["score_big_deep_pre"].values

        df_batch["score_coelution_center_sub_left"] = sa_center - sa_left
        df_batch["score_deep_center_sub_left"] = deep_center - deep_left
        df_batch["score_coelution_x_center"] = sa_center * deep_center
        df_batch["score_coelution_x_big"] = sa_center * deep_big
    else:
        # refine model + non-ppm
        sa_center = df_batch["score_center_coelution"].values
        deep_center = df_batch["score_center_deep_refine"].values
        deep_left = df_batch["score_left_deep_refine"].values
        deep_big = df_batch["score_big_deep_refine"].values

        df_batch["score_deep_center_sub_left_refine"] = deep_center - deep_left
        df_batch["score_coelution_x_center_refine"] = sa_center * deep_center
        df_batch["score_coelution_x_big_refine"] = sa_center * deep_big

    return df_batch


def update_scores(
    df: pd.DataFrame,
    ms: tims.Tims,
    model_center: torch.nn.Module,
    model_big: torch.nn.Module,
    model_mall: torch.nn.Module,
) -> pd.DataFrame:
    """
    Calculate scores using the refined DeepProfile and the trained DeepMall.
        1. DeepProfile: refined deep prob scores
        2. DeepProfile: cross scores with refined deep prob scores
        3. DeepProfile: refined deep prob and layer scores with 0.5 * ppm
        4. DeepProfile: refined deep prob and layer scores with 0.25 * ppm
        5. DeepMall: deep prob and layer scores
    """
    df_good = []
    for swath_id in df["swath_id"].unique():
        df_swath = df[df["swath_id"] == swath_id]
        df_swath = df_swath.reset_index(drop=True)

        # map_gpu
        ms1_profile, ms2_profile = ms.copy_map_to_gpu(swath_id, centroid=False)
        ms1_centroid, ms2_centroid = ms.copy_map_to_gpu(swath_id, centroid=True)

        batch_n = cfg.batch_deep_big

        for _, df_batch in df_swath.groupby(df_swath.index // batch_n):
            df_batch = df_batch.reset_index(drop=True)
            # deepmap-refined scores without feature
            scores_deep_v, _ = deepmap.extract_scoring_big(
                model_center,
                model_big,
                df_batch,
                ms1_profile,
                ms2_profile,
                cfg.map_cycle_dim,
                cfg.map_im_gap,
                cfg.map_im_dim,
                cfg.tol_ppm,
                cfg.tol_im_map,
            )
            df_batch = scoring_by_deep_prob(df_batch, scores_deep_v, x="refine")
            df_batch = scoring_by_cross(df_batch, is_update=True)

            # 0.5*ppm
            scores_deep_v, features_deep_v = deepmap.extract_scoring_big(
                model_center,
                model_big,
                df_batch,
                ms1_profile,
                ms2_profile,
                cfg.map_cycle_dim,
                cfg.map_im_gap,
                cfg.map_im_dim,
                cfg.tol_ppm * 0.5,
                cfg.tol_im_map,
            )
            df_batch = scoring_by_deep_prob(df_batch, scores_deep_v, x="refine_p1")
            df_batch = scoring_by_deep_layer(df_batch, features_deep_v, x="refine_p1")

            # 0.25*ppm
            scores_deep_v, features_deep_v = deepmap.extract_scoring_big(
                model_center,
                model_big,
                df_batch,
                ms1_profile,
                ms2_profile,
                cfg.map_cycle_dim,
                cfg.map_im_gap,
                cfg.map_im_dim,
                cfg.tol_ppm * 0.25,
                cfg.tol_im_map,
            )
            df_batch = scoring_by_deep_prob(df_batch, scores_deep_v, x="refine_p2")
            df_batch = scoring_by_deep_layer(df_batch, features_deep_v, x="refine_p2")

            # deepmall
            scores_mall, features_mall = deepmall.scoring_mall(
                model_mall,
                df_batch,
                ms1_centroid,
                ms2_centroid,
                cfg.tol_im_xic,
                cfg.tol_ppm,
            )
            df_batch["score_mall"] = scores_mall

            m = features_mall.shape[-1]
            columns = ["score_ft_mall_" + str(i) for i in range(m)]
            df_batch[columns] = features_mall

            df_good.append(df_batch)

        utils.release_gpu_scans(ms1_profile, ms2_profile, ms1_centroid, ms2_centroid)

    df = pd.concat(df_good, axis=0, ignore_index=True)
    utils.cal_acc_recall(cfg.ws_single, df[df["decoy"] == 0], diann_q_pr=0.01)

    return df
