import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numba import cuda

from full_dia import cfg, models, utils
from full_dia.utils import create_cuda_zeros

try:
    _ = profile
except NameError:

    def profile(func):
        return func


def load_models(dir_center=None, dir_big=None):
    """
    Load DeepProfile-14 and DeepProfile-56 models.
    """
    channels = 2 + cfg.fg_num
    model_center = load_model_center(dir_center, channels)

    channels = 4 * (2 + cfg.fg_num)
    model_big = load_model_big(dir_big, channels)

    return model_center, model_big


def load_model_big(dir_model, n_channel):
    """
    Load DeepProfile-56 model.
    """
    model = models.DeepMap(n_channel)
    device = cfg.gpu_id
    if dir_model is None:
        pt_path = Path(__file__).resolve().parent / "pretrained" / "deepbig_ys_fast.pt"
        model.load_state_dict(torch.load(pt_path, map_location=device))
    else:
        model.load_state_dict(torch.load(dir_model, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def load_model_center(dir_model, n_channel):
    """
    Load DeepProfile-14 model.
    """
    model = models.DeepMap(n_channel)
    device = cfg.gpu_id
    if dir_model is None:
        pt_path = (
            Path(__file__).resolve().parent / "pretrained" / "deepcenter_ys_fast.pt"
        )
        model.load_state_dict(torch.load(pt_path, map_location=device))
    else:
        model.load_state_dict(torch.load(dir_model, map_location=device))
    model = model.to(device)
    model.eval()
    return model


@cuda.jit(device=True)
def find_first_index(scan_mz, query_left, query_right):
    """
    Find first index that match the query value

    Parameters
    ----------
    scan_mz : cuda.array
        MS data of a cycle with m/z ascending order.

    query_left : float
        The target m/z value with - ppm.

    query_right : float
        The target m/z value with + ppm.

    Returns
    -------
    best_j : int
        The index of the first m/z that matches the query.
    """
    scan_len = len(scan_mz)

    low = 0
    high = scan_len - 1
    best_j = 0
    if scan_mz[low] == query_left:
        best_j = low
    elif scan_mz[high] == query_right:
        best_j = high
    else:
        while high - low > 1:
            mid = (low + high) // 2
            if scan_mz[mid] == query_left:
                best_j = mid
                break
            if scan_mz[mid] < query_left:
                low = mid
            else:
                high = mid
        if best_j == 0:  # on match，high-low=1
            if abs(scan_mz[low] - query_left) < abs(scan_mz[high] - query_left):
                best_j = low
            else:
                best_j = high
    # find first match in list!
    while best_j > 0:
        if scan_mz[best_j - 1] == scan_mz[best_j]:
            best_j = best_j - 1
        else:
            break

    return best_j


@cuda.jit
def gpu_bin_map(
    n,
    cycle_num,
    idx_start_v,
    ms1_scan_seek_idx,
    ms1_scan_im,
    ms1_scan_mz,
    ms1_scan_height,
    ms2_scan_seek_idx,
    ms2_scan_im,
    ms2_scan_mz,
    ms2_scan_height,
    query_mz_m,
    ppm_tolerance,
    query_im_v,
    im_tolerance,
    im_gap,
    result_maps,
    ms1_ion_num,
):
    """
    Each CUDA thread generates a map (cycle + mobility + intensity) of an elution group.
    When multiple signals fall into the same bin, retain only the one with the highest intensity.
    """
    thread_idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if thread_idx >= n:
        return

    # pr idx, ion idx
    ions_num = query_mz_m.shape[1]
    k = thread_idx // ions_num
    ion_idx = thread_idx % ions_num

    # params
    query_mz = query_mz_m[k, ion_idx]
    query_mz_left = query_mz * (1.0 - ppm_tolerance / 1000000.0)
    query_mz_right = query_mz * (1.0 + ppm_tolerance / 1000000.0)
    query_im = query_im_v[k]
    query_im_left = query_im - im_tolerance
    query_im_right = query_im + im_tolerance
    im_base = query_im - im_tolerance

    # both for ms1 and ms2
    idx_start = idx_start_v[k]
    idx_end = idx_start + cycle_num

    if ion_idx < ms1_ion_num:
        scans_seek_idx = ms1_scan_seek_idx
        scans_im = ms1_scan_im
        scans_mz = ms1_scan_mz
        scans_height = ms1_scan_height
    else:
        scans_seek_idx = ms2_scan_seek_idx
        scans_im = ms2_scan_im
        scans_mz = ms2_scan_mz
        scans_height = ms2_scan_height
    for cycle_idx, scan_idx in enumerate(range(idx_start, idx_end)):
        start = scans_seek_idx[scan_idx]
        end = scans_seek_idx[scan_idx + 1]
        scan_len = end - start
        scan_im = scans_im[start:end]
        scan_mz = scans_mz[start:end]
        scan_height = scans_height[start:end]

        seek = find_first_index(scan_mz, query_mz_left, query_mz_right)

        while seek < scan_len:
            mz = scan_mz[seek]
            if mz > query_mz_right:
                break
            elif mz < query_mz_left:  # exist multiple mz values
                seek += 1
                continue
            else:
                im = scan_im[seek]
                if query_im_left < im < query_im_right:
                    y = scan_height[seek]
                    im_idx = int((im - im_base) / im_gap)
                    y_map_curr = result_maps[k, ion_idx, cycle_idx, im_idx]
                    if y > y_map_curr:
                        result_maps[k, ion_idx, cycle_idx, im_idx] = y
                seek += 1


@cuda.jit
def gpu_bin_maps(
    n,
    locus_num,
    cycle_num,
    idx_start_m,
    ms1_scan_seek_idx,
    ms1_scan_im,
    ms1_scan_mz,
    ms1_scan_height,
    ms2_scan_seek_idx,
    ms2_scan_im,
    ms2_scan_mz,
    ms2_scan_height,
    query_mz_m,
    tol_ppm,
    query_im_v,
    tol_im_map,
    im_gap,
    result_maps,
    ms1_ion_num,
):
    """
    maps: [n_pr, n_locus, n_ion, n_cycle, n_im_bin]
    Each thread generates maps for multi elution groups of a pr
    When multiple signals fall into the same bin, retain only the one with the highest intensity.
    """
    thread_idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if thread_idx >= n:
        return

    # pr idx, ion idx
    ions_num = query_mz_m.shape[1]
    locus_per_peptide = locus_num * ions_num
    k = thread_idx // locus_per_peptide
    locus = thread_idx % locus_per_peptide // ions_num
    ion_idx = thread_idx % locus_per_peptide % ions_num

    # params
    query_mz = query_mz_m[k, ion_idx]
    query_mz_left = query_mz * (1.0 - tol_ppm / 1000000.0)
    query_mz_right = query_mz * (1.0 + tol_ppm / 1000000.0)
    query_im = query_im_v[k]
    query_im_left = query_im - tol_im_map
    query_im_right = query_im + tol_im_map
    im_base = query_im - tol_im_map

    ## both for ms1 and ms2
    idx_start = idx_start_m[k, locus]
    idx_end = idx_start + cycle_num

    if ion_idx < ms1_ion_num:
        scans_seek_idx = ms1_scan_seek_idx
        scans_im = ms1_scan_im
        scans_mz = ms1_scan_mz
        scans_height = ms1_scan_height
    else:
        scans_seek_idx = ms2_scan_seek_idx
        scans_im = ms2_scan_im
        scans_mz = ms2_scan_mz
        scans_height = ms2_scan_height

    for cycle_idx, scan_idx in enumerate(range(idx_start, idx_end)):
        start = scans_seek_idx[scan_idx]
        end = scans_seek_idx[scan_idx + 1]
        scan_len = end - start
        scan_im = scans_im[start:end]
        scan_mz = scans_mz[start:end]
        scan_height = scans_height[start:end]

        seek = find_first_index(scan_mz, query_mz_left, query_mz_right)
        while seek < scan_len:
            x = scan_mz[seek]
            if x > query_mz_right:
                break
            elif x < query_mz_left:  # exist multi mz values
                seek += 1
                continue
            else:
                im = scan_im[seek]
                if query_im_left < im < query_im_right:
                    y = scan_height[seek]
                    im_idx = int((im - im_base) / im_gap)
                    y_curr = result_maps[k, locus, ion_idx, cycle_idx, im_idx]
                    if y > y_curr:
                        result_maps[k, locus, ion_idx, cycle_idx, im_idx] = y
                seek += 1


@profile
def extract_maps(
    df_batch: pd.DataFrame,
    idx_start_m: np.ndarray,
    locus_num: int,
    cycle_num: int,
    map_im_size: int,
    map_gpu_ms1: dict,
    map_gpu_ms2: dict,
    tol_ppm: float,
    tol_im_map: float,
    im_gap: float,
    neutron_num: int,
) -> torch.Tensor:
    """
    Extrac maps for multi elution groups of a pr.

    Parameters
    ----------
    df_batch : pd.DataFrame
        Provide pr info.

    idx_start_m : np.ndarray
        Cycle start index.

    locus_num : int
        How many locus to extract for a pr.

    cycle_num : int
        How many cycle to extract for a pr. Default 13.

    map_im_size : float
        Default 50.

    map_gpu_ms1 : dict
        Provide the MS1 data.

    map_gpu_ms2 : dict
        Provide the MS2 data.

    tol_ppm : float
        The tolerance of ppm.

    tol_im_map : float
        The tolerance of im.

    im_gap : float
        The bin width of im in a map.

    neutron_num : int
        Specify the neutron num.

    Returns
    -------
    Maps : torch.Tensor
        The extracted maps.
    """
    batch_size = len(df_batch)

    idx_start = idx_start_m[df_batch.index]

    # params
    if neutron_num == -1:
        query_mz_ms1 = df_batch["pr_mz_left"].values
        query_mz_ms1 = np.tile(query_mz_ms1, (2, 1)).T
        query_mz_ms2 = np.array(df_batch["fg_mz_left"].values.tolist())
        query_mz_m = np.concatenate([query_mz_ms1, query_mz_ms2], axis=1)
        ms1_ion_num = 1
    elif neutron_num == 0:
        query_mz_ms1 = df_batch["pr_mz"].values
        query_mz_ms1 = np.tile(query_mz_ms1, (2, 1)).T
        cols_center = ["fg_mz_" + str(i) for i in range(cfg.fg_num)]
        query_mz_ms2 = df_batch[cols_center].values
        query_mz_m = np.concatenate([query_mz_ms1, query_mz_ms2], axis=1)
        ms1_ion_num = 1
    elif neutron_num == 1:
        query_mz_ms1 = df_batch["pr_mz_1H"].values
        query_mz_ms1 = np.tile(query_mz_ms1, (2, 1)).T
        cols_1H = ["fg_mz_1H_" + str(i) for i in range(cfg.fg_num)]
        query_mz_ms2 = df_batch[cols_1H].values
        query_mz_m = np.concatenate([query_mz_ms1, query_mz_ms2], axis=1)
        ms1_ion_num = 1
    elif neutron_num == 2:
        query_mz_ms1 = df_batch["pr_mz_2H"].values
        query_mz_ms1 = np.tile(query_mz_ms1, (2, 1)).T
        cols_2H = ["fg_mz_2H_" + str(i) for i in range(cfg.fg_num)]
        query_mz_ms2 = df_batch[cols_2H].values
        query_mz_m = np.concatenate([query_mz_ms1, query_mz_ms2], axis=1)
        ms1_ion_num = 1
    elif neutron_num > 2:  # total
        ms1_cols = [
            "pr_mz_left",
            "pr_mz",
            "pr_mz_1H",
            "pr_mz_2H",
            "pr_mz_left",
            "pr_mz",
            "pr_mz_1H",
            "pr_mz_2H",
        ]  # unfrag
        ms1 = df_batch[ms1_cols].values
        cols_left = ["fg_mz_left_" + str(i) for i in range(cfg.fg_num)]
        left = df_batch[cols_left].values
        cols_center = ["fg_mz_" + str(i) for i in range(cfg.fg_num)]
        center = df_batch[cols_center].values
        cols_1H = ["fg_mz_1H_" + str(i) for i in range(cfg.fg_num)]
        fg_1H = df_batch[cols_1H].values
        cols_2H = ["fg_mz_2H_" + str(i) for i in range(cfg.fg_num)]
        fg_2H = df_batch[cols_2H].values
        query_mz_m = np.concatenate([ms1, left, center, fg_1H, fg_2H], axis=1)
        ms1_ion_num = 4
    else:
        raise ValueError("neutron_num in extract_maps has to be [-1, 1, 2, >2]")

    query_im_v = df_batch["measure_im"].values

    # GPU
    idx_start = cuda.to_device(idx_start)
    query_mz_m = cuda.to_device(query_mz_m)
    query_im_v = cuda.to_device(query_im_v)
    result_maps = create_cuda_zeros(
        (batch_size, locus_num, query_mz_m.shape[1], cycle_num, map_im_size)
    )
    # kernel func, each thread generates maps for a pr
    k = batch_size
    n = k * locus_num * query_mz_m.shape[1]
    threads_per_block = 512
    blocks_per_grid = math.ceil(n / threads_per_block)
    gpu_bin_maps[blocks_per_grid, threads_per_block](
        n,
        locus_num,
        cycle_num,
        idx_start,
        map_gpu_ms1["scan_seek_idx"],
        map_gpu_ms1["scan_im"],
        map_gpu_ms1["scan_mz"],
        map_gpu_ms1["scan_height"],
        map_gpu_ms2["scan_seek_idx"],
        map_gpu_ms2["scan_im"],
        map_gpu_ms2["scan_mz"],
        map_gpu_ms2["scan_height"],
        query_mz_m,
        tol_ppm,
        query_im_v,
        tol_im_map,
        im_gap,
        result_maps,
        ms1_ion_num,
    )
    cuda.synchronize()

    result_maps = utils.convert_numba_to_tensor(result_maps)
    return result_maps


@profile
def scoring_maps(
    model: torch.nn.Module,
    df_input: pd.DataFrame,
    map_gpu_ms1: dict,
    map_gpu_ms2: dict,
    cycle_num: int,
    map_im_gap: float,
    map_im_dim: float,
    ppm_tolerance: float,
    im_tolerance: float,
    neutron_num: int,
    return_feature: bool = True,
) -> tuple:
    """
    Extract and score the Profile-14 maps.

    Parameters
    ----------
    model : torch.nn.Module
        DeepProfile-14

    df_input : pd.DataFrame
        Provide the pr info.

    map_gpu_ms1 : dict
        Provide the MS1 data.

    map_gpu_ms2 : dict
        Provide the MS2 data.

    cycle_num : int
        How many cycle to extract for a pr. Default 13.

    map_im_gap : float
        The bin width of im in a map.

    map_im_dim : float
        The dimension of im in a map.

    ppm_tolerance : float
        The tolerance of ppm.

    im_tolerance : float
        The tolerance of im.

    neutron_num : int
        Specify the neutron num.

    return_feature : bool, default=True
        Whether to return feature or not.

    Returns
    -------
    tuple
        pred : torch.Tensor
            The scores by DeepProfile-14.

        features : np.ndarray
            The features by DeepProfile-14.
    """
    # locus
    locus_m = df_input["locus"].values.reshape(-1, 1)
    locus_num = locus_m.shape[1]

    # cycle start and end
    cycle_total = len(map_gpu_ms1["scan_rts"])
    idx_start_m = locus_m - int((cycle_num - 1) / 2)
    idx_start_m[idx_start_m < 0] = 0
    idx_start_max = cycle_total - cycle_num
    idx_start_m[idx_start_m > idx_start_max] = idx_start_max

    # in batches
    feature_v, pred_v = [], []
    batch_num = cfg.batch_deep_center
    for _, df_batch in df_input.groupby(df_input.index // batch_num):
        maps = extract_maps(
            df_batch,
            idx_start_m,
            locus_num,
            cycle_num,
            map_im_dim,
            map_gpu_ms1,
            map_gpu_ms2,
            ppm_tolerance,
            im_tolerance,
            map_im_gap,
            neutron_num=neutron_num,
        )
        # maps: [k, locus, 2+fg_num, map_cycle_dim, map_im_dim]
        maps = maps.view(
            maps.shape[0] * maps.shape[1], maps.shape[2], maps.shape[3], maps.shape[4]
        )
        # valid ion nums
        non_fg_num = maps.shape[1] - cfg.fg_num
        valid_ion_nums = non_fg_num + df_batch["fg_num"].values
        valid_ion_nums = (
            torch.from_numpy(np.repeat(valid_ion_nums, locus_num)).long().to(cfg.gpu_id)
        )
        with torch.no_grad():
            # with torch.cuda.amp.autocast():
            feature, pred = model(maps, valid_ion_nums)
        torch.cuda.synchronize()  # for profile

        pred = torch.softmax(pred, 1)
        pred = pred[:, 1].view(len(df_batch), locus_num)
        pred_v.append(pred)

        if return_feature:
            feature = feature.view(len(df_batch), locus_num, -1)
            feature = feature.cpu()
            feature = feature.numpy()
            feature_v.append(feature)

    pred = torch.cat(pred_v).to(dtype=torch.float32)  # torch autocast to 16
    feature = np.vstack(feature_v) if return_feature else None

    return pred, feature


@profile
def extract_scoring_big(
    model_center: torch.nn.Module,
    model_big: torch.nn.Module,
    df_input: pd.DataFrame,
    map_gpu_ms1: dict,
    map_gpu_ms2: dict,
    cycle_num: int,
    map_im_gap: float,
    map_im_dim: float,
    ppm_tolerance: float,
    im_tolerance: float,
) -> tuple:
    """
    Extrac and scoring Maps using DeepProfile-14 and DeepProfile-56.

    Parameters
    ----------
    model_center : torch.nn.Module
        The DeepProfile-14 model.

    model_big : torch.nn.Module
        The DeepProfile-56 model.

    df_input : pd.DataFrame
        Provide the pr info.

    map_gpu_ms1 : dict
        Provide the MS1 data.

    map_gpu_ms2 : dict
        Provide the MS2 data.

    cycle_num : int
        How many cycle to extract for a pr. Default 13.

    map_im_gap : float
        The bin width of im in a map.

    map_im_dim : float
        The dimension of im in a map.

    ppm_tolerance : float
        The ppm tolerance.

    im_tolerance : float
        The mobility tolerance.

    Returns
    -------
    tuple
        pred_v : list[np.ndarray]
            The deep scores for [14-left, 14-center, 14-1H, 14-2H, 56-total]

        feature_v : list[np.ndarray]
            The deep features for [14-left, 14-center, 14-1H, 14-2H, 56-total]
    """
    # locus
    locus_v = df_input["locus"].values

    # cycle start and end
    cycle_total = len(map_gpu_ms1["scan_rts"])
    idx_start_v = locus_v - int((cycle_num - 1) / 2)
    idx_start_v[idx_start_v < 0] = 0
    idx_start_max = cycle_total - cycle_num
    idx_start_v[idx_start_v > idx_start_max] = idx_start_max

    # params
    ms1_cols = [
        "pr_mz_left",
        "pr_mz",
        "pr_mz_1H",
        "pr_mz_2H",
        "pr_mz_left",
        "pr_mz",
        "pr_mz_1H",
        "pr_mz_2H",
    ]  # unfrag
    ms1 = df_input[ms1_cols].values
    cols_left = ["fg_mz_left_" + str(i) for i in range(cfg.fg_num)]
    left = df_input[cols_left].values
    cols_center = ["fg_mz_" + str(i) for i in range(cfg.fg_num)]
    center = df_input[cols_center].values
    cols_1H = ["fg_mz_1H_" + str(i) for i in range(cfg.fg_num)]
    fg_1H = df_input[cols_1H].values
    cols_2H = ["fg_mz_2H_" + str(i) for i in range(cfg.fg_num)]
    fg_2H = df_input[cols_2H].values
    query_mz_m = np.concatenate([ms1, left, center, fg_1H, fg_2H], axis=1)
    ms1_ion_num = 4

    query_im_v = df_input["measure_im"].values

    # cuda input
    idx_start_v = cuda.to_device(idx_start_v)
    query_mz_m = cuda.to_device(query_mz_m)
    query_im_v = cuda.to_device(query_im_v)

    # cuda output
    n = len(df_input)
    ions_num = query_mz_m.shape[1]
    maps = create_cuda_zeros((n, ions_num, cycle_num, map_im_dim))

    # kernel func, each thread for a elution groups of a pr
    thread_num = n * ions_num
    threads_per_block = 256
    blocks_per_grid = math.ceil(thread_num / threads_per_block)
    gpu_bin_map[blocks_per_grid, threads_per_block](
        thread_num,
        cycle_num,
        idx_start_v,
        map_gpu_ms1["scan_seek_idx"],
        map_gpu_ms1["scan_im"],
        map_gpu_ms1["scan_mz"],
        map_gpu_ms1["scan_height"],
        map_gpu_ms2["scan_seek_idx"],
        map_gpu_ms2["scan_im"],
        map_gpu_ms2["scan_mz"],
        map_gpu_ms2["scan_height"],
        query_mz_m,
        ppm_tolerance,
        query_im_v,
        im_tolerance,
        map_im_gap,
        maps,
        ms1_ion_num,
    )
    cuda.synchronize()

    # -1H, center, +H, +2H, total
    maps = utils.convert_numba_to_tensor(maps)

    pred_v, feature_v = [], []
    for i in range(5):
        if i != 4:
            idx = [i, i + 4] + list(range(8 + i * 12, 20 + i * 12))
            valid_ion_nums = 2 + df_input["fg_num"].values
            model = model_center
        else:
            idx = list(range(56))
            valid_ion_nums = 4 * (2 + df_input["fg_num"].values)
            model = model_big
        maps_sub = maps[:, idx]
        valid_ion_nums = torch.from_numpy(valid_ion_nums).long().to(cfg.gpu_id)
        with torch.no_grad():
            # with torch.cuda.amp.autocast():
            feature, pred = model(maps_sub, valid_ion_nums)
        torch.cuda.synchronize()
        pred = torch.softmax(pred, 1)
        pred = pred[:, 1].cpu().numpy().astype(np.float32)
        feature = feature.cpu().numpy()
        pred_v.append(pred)
        feature_v.append(feature)

    return pred_v, feature_v
