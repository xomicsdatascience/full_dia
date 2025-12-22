import copy

import numpy as np
import pandas as pd
import torch

from full_dia import cfg, dataloader, deepmall, deepmap, fxic, models, tims, utils
from full_dia.log import Logger

logger = Logger.get_logger()

try:
    _ = profile
except NameError:

    def profile(func):
        return func


@profile
def construct_train_data(df_top: pd.DataFrame, ms: tims.Tims) -> tuple:
    """
    Construct maps and mall data.
    Positive samples: [Apex, Apex + 1, Apex - 1] locus from target peak groups.
    Negative samples: top-3 (by SA score) locus from target peak groups.

    Parameters
    ----------
    df_top : pd.DataFrame
        Provide the identification information.

    ms : tims.Tims
        MS data.

    Returns
    -------
    tuple
        maps_center : np.ndarray
            The maps data for monoisotope ions. Dimension: [n_sample, 14, n_cycle, n_im_bin].

        maps_big : np.ndarray
            The maps data for monoisotope + isotope ions. Dimension: [n_sample, 56, n_cycle, n_im_bin].

        malls : np.ndarray
            The mall data for the calculation of intensity similarity.

        center_ion_nums : np.ndarray
            Valid ions num for each sample.

        labels : np.ndarray
            Positive or negative.
    """
    # targets within FDR-%1 are pos samples
    df_target = df_top[
        (df_top["decoy"] == 0)
        & (df_top["group_rank"] == 1)
        & (df_top["q_pr_run"] < 0.01)
    ].reset_index(drop=True)
    if len(df_target) > 10000:
        df_target = df_target.sample(n=10000, random_state=1, replace=False)

    # find sub-best elution groups in the range of whole gradient
    # sub-best elution groups are neg samples
    locus_v = []
    measure_ims_v = []
    df_v = []
    for swath_id in df_target["swath_id"].unique():
        df_swath = df_target[df_target["swath_id"] == swath_id]
        df_swath = df_swath.reset_index(drop=True)

        # map_gpu
        ms1_profile, ms2_profile = ms.copy_map_to_gpu(swath_id, centroid=False)
        ms1_centroid, ms2_centroid = ms.copy_map_to_gpu(swath_id, centroid=True)
        N = 10000

        for _, df_batch in df_swath.groupby(df_swath.index // N):
            df_batch = df_batch.reset_index(drop=True)
            # [k, ions_num, n]，the range of whole gradient
            locus, rts, ims, mzs, xics = fxic.extract_xics(
                df_batch,
                ms1_centroid,
                ms2_centroid,
                cfg.tol_ppm,
                cfg.tol_im_xic,
            )
            xics = fxic.gpu_simple_smooth(xics)
            scores_sa, scores_sa_m = fxic.cal_coelution_by_gaussion(
                xics, cfg.window_points, df_batch.fg_num.values + 2
            )
            scores_sa_gpu = fxic.reserve_sa_maximum(scores_sa)
            _, idx = torch.topk(scores_sa_gpu, k=30, dim=1, sorted=True)
            locus = locus[np.arange(len(locus))[:, None], idx.cpu()]
            locus_v.append(locus)
            df_v.append(df_batch)

            # cal measure_im for maps by locus
            n_pep, n_ion, n_cycle = ims.shape
            ims = ims.transpose(0, 2, 1).reshape(-1, n_ion)
            scores_sa_m = scores_sa_m.cpu().numpy()
            scores_sa_m = scores_sa_m.transpose(0, 2, 1).reshape(-1, n_ion)
            measure_ims = fxic.cal_measure_im(ims, scores_sa_m)
            measure_ims = measure_ims.reshape(-1, n_cycle)
            measure_ims_v.append(measure_ims)

    locus_neg_m = np.vstack(locus_v)
    df_target = pd.concat(df_v, ignore_index=True)
    measure_ims = np.vstack(measure_ims_v)

    # pos: apex and ±1 cycle for data augmentation.
    locus_pos = df_target["locus"].values
    df_target["decoy"] = 0
    idx_x = np.arange(len(df_target))
    # target_ims = measure_ims[idx_x, locus_pos]
    # assert np.abs(df_target['measure_im'] - target_ims).max() < 0.02

    df_target_left = df_target.copy()
    df_target_left["locus"] = df_target_left["locus"] - 1
    df_target_right = df_target.copy()
    df_target_right["locus"] = df_target_right["locus"] + 1
    df_targets = pd.concat([df_target, df_target_left, df_target_right])
    df_targets = df_targets.reset_index(drop=True)
    data_augment_num = int(len(df_targets) / len(df_target))

    # putative must >7 cycles
    for i in range(locus_neg_m.shape[1]):
        locus = locus_neg_m[:, i]
        good_idx = np.abs(locus - locus_pos) > 7
        locus_neg_m[:, i][~good_idx] = 0
    locus_neg_m = utils.move_all_zeros_end(locus_neg_m)
    locus_neg_m = locus_neg_m[:, :data_augment_num]
    assert (locus_neg_m >= 0).all()

    # concat pos and neg to df
    df_v = []
    for i in range(locus_neg_m.shape[1]):
        df = df_target.copy()
        df["locus"] = locus_neg_m[:, i]
        df["measure_im"] = measure_ims[idx_x, locus_neg_m[:, i]]
        df["decoy"] = 1
        df_v.append(df)
    df_negs = pd.concat(df_v, axis=0, ignore_index=True)
    df = pd.concat([df_targets, df_negs], axis=0, ignore_index=True)
    locus_m = df["locus"].values.reshape(-1, 1)

    # extract map
    cycle_total = len(ms1_profile["scan_rts"])
    cycle_num = cfg.map_cycle_dim
    idx_start_bank = locus_m - int((cycle_num - 1) / 2)
    idx_start_bank[idx_start_bank < 0] = 0
    idx_start_max = cycle_total - cycle_num
    idx_start_bank[idx_start_bank > idx_start_max] = idx_start_max

    maps_center_v, maps_big_v, mall_v, ion_nums_v, labels_v = [], [], [], [], []
    for swath_id in df["swath_id"].unique():
        ms1_profile, ms2_profile = ms.copy_map_to_gpu(swath_id, centroid=False)
        ms2_centroid, ms2_centroid = ms.copy_map_to_gpu(swath_id, centroid=True)

        df_swath = df[df["swath_id"] == swath_id]
        idx_start_m = idx_start_bank[df_swath.index]
        df_swath = df_swath.reset_index(drop=True)

        for _, df_batch in df_swath.groupby(df_swath.index // 1000):
            ion_nums = 2 + df_batch["fg_num"].values
            ion_nums_v.append(ion_nums)
            labels_v.append(1 - df_batch["decoy"].values)
            maps_big = deepmap.extract_maps(
                df_batch,
                idx_start_m,
                locus_m.shape[1],
                cycle_num,
                cfg.map_im_dim,
                ms1_profile,
                ms2_profile,
                cfg.tol_ppm,
                cfg.tol_im_map,
                cfg.map_im_gap,
                neutron_num=100,
            )  # big
            maps_big = maps_big.squeeze(dim=1).cpu().numpy()
            cols_idx = [1, 5] + list(range(20, 32))
            maps_center = maps_big[:, cols_idx]
            maps_center_v.append(maps_center)
            maps_big_v.append(maps_big)

            mall = deepmall.extract_mall(
                df_batch,
                ms1_centroid,
                ms2_centroid,
                cfg.tol_im_xic,
                cfg.tol_ppm,
            )
            mall_v.append(mall.cpu().numpy())
    utils.release_gpu_scans(ms1_profile, ms2_profile)

    maps_center = np.vstack(maps_center_v)
    maps_big = np.vstack(maps_big_v)
    malls = np.vstack(mall_v)
    center_ion_nums = np.concatenate(ion_nums_v, dtype=np.int8)
    labels = np.concatenate(labels_v)

    return maps_center, maps_big, malls, center_ion_nums, labels


def make_dataset_maps(
    maps: np.ndarray,
    valid_num: np.ndarray,
    labels: np.ndarray,
    train_ratio: float,
    maps_type: str,
) -> tuple:
    """
    Make pytorch dataset and split it into train and validation sets for Map data.

    Parameters
    ----------
    maps : np.ndarray
        The map/profile data.

    valid_num : np.ndarray
        Valid ion num of each map.

    labels : np.ndarray
        The labels.

    train_ratio : float
        The ratio between train set and validation set.

    maps_type : str
        "Profile-14": for 14 monoisotope ions (pr, pr_unfrag, 12 fragment ions)
        "Profile-56": for monoisotope + isotope ions (14 * 4)

    Returns
    -------
    tuple
        train : torch.utils.data.Dataset
        eval : torch.utils.data.Dataset
    """
    dataset = dataloader.MapDataset(maps, valid_num, labels)
    train_num = int(train_ratio * len(dataset))
    eval_num = len(dataset) - train_num
    train, eval = torch.utils.data.random_split(
        dataset, [train_num, eval_num], generator=torch.Generator().manual_seed(123)
    )
    info = "Deep{} refine with train: {}, eval: {}".format(
        maps_type, len(train), len(eval)
    )
    logger.info(info)

    return train, eval


def make_dataset_mall(
    malls: np.ndarray,
    valid_num: np.ndarray,
    labels: np.ndarray,
    train_ratio: float = 0.9,
) -> tuple:
    """
    Make pytorch dataset and split it into train and validation sets for Mall data.

    Parameters
    ----------
    malls : np.ndarray
        The mall data.

    valid_num : np.ndarray
        Valid ion num of each mall.

    labels : np.ndarray
        The labels.

    train_ratio : float, default=0.9
        The ratio between train set and validation set.

    Returns
    -------
    tuple
        train : torch.utils.data.Dataset
        eval : torch.utils.data.Dataset
        Mall's feature dimention.
    """
    dataset = dataloader.MallDataset(malls, valid_num, labels)
    train_num = int(train_ratio * len(dataset))
    eval_num = len(dataset) - train_num
    train, eval = torch.utils.data.random_split(
        dataset, [train_num, eval_num], generator=torch.Generator().manual_seed(123)
    )
    info = "DeepMall train with train: {}, eval: {}".format(len(train), len(eval))
    logger.info(info)

    return train, eval, malls.shape[1]


def my_collate(items):
    """
    The recall function of pytorch dataloader.
    """
    maps, valid_nums, labels = zip(*items)

    xic = torch.from_numpy(np.array(maps))
    xic_num = torch.tensor(valid_nums)
    label = torch.tensor(labels)

    return xic, xic_num, label


def eval_one_epoch(
    trainloader: torch.utils.data.DataLoader, model: torch.nn.Module
) -> float:
    """
    Return the accuracy of the model on the validation set.
    """
    device = cfg.gpu_id
    model.eval()
    prob_v, label_v = [], []

    for _, (batch_map, batch_map_len, batch_y) in enumerate(trainloader):
        batch_map = batch_map.float().to(device)
        batch_map_len = batch_map_len.long().to(device)
        batch_y = batch_y.long().to(device)

        # forward
        with torch.no_grad():
            features, prob = model(batch_map, batch_map_len)
        prob = torch.softmax(prob.view(-1, 2), 1)
        prob = prob[:, 1].tolist()

        prob_v.extend(prob)
        label_v.extend(batch_y.cpu().tolist())

    prob_v = np.array(prob_v)
    label_v = np.array(label_v)

    # acc
    prob_v[prob_v >= 0.5] = 1
    prob_v[prob_v < 0.5] = 0
    acc = sum(prob_v == label_v) / len(label_v)
    # recall = sum(prob_v[label_v == 1] == 1) / sum(label_v == 1)
    # fscore = 2 * acc * recall / (acc + recall)
    return acc


def train_one_epoch(
    trainloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
) -> float:
    """
    Train the model on the training set and return the loss.
    """
    device = cfg.gpu_id
    model.train()
    epoch_loss = 0.0
    for _, (batch_map, batch_map_len, batch_y) in enumerate(trainloader):
        batch_map = batch_map.float().to(device)
        batch_map_len = batch_map_len.long().to(device)
        batch_y = batch_y.long().to(device)

        # forward
        features, batch_pred = model(batch_map, batch_map_len)

        # loss
        batch_loss = loss_fn(batch_pred, batch_y)

        # back
        optimizer.zero_grad()
        batch_loss.backward()
        # update
        optimizer.step()

        # log
        epoch_loss += batch_loss.item()

    epoch_loss = epoch_loss / len(trainloader)
    return epoch_loss


def retrain_model_map(
    model_maps: torch.nn.Module,
    maps: np.ndarray,
    valid_nums: np.ndarray,
    labels: np.ndarray,
    maps_type: str,
    epochs: int,
) -> torch.nn.Module:
    """
    Fine-tune the model and return the model with optimal performance.

    Parameters
    ----------
    model_maps : torch.nn.Module
        The pretrained DeepProfile model.

    maps : np.ndarray
        Run-specific profile/map data for fine-tuning.

    valid_nums : np.ndarray
        Valid ion num of each train sample.

    labels : np.ndarray
        The labels of train samples.

    maps_type : str
        "Profile-14": for 14 monoisotope ions (pr, pr_unfrag, 12 fragment ions)
        "Profile-56": for monoisotope + isotope ions (14 * 4)

    epochs : int
        Number of maximum epochs.

    Returns
    -------
    model_best : torch.nn.Module
        The model with optimal performance.
    """
    batch_size = 64
    num_workers = 0

    train_dataset, eval_dataset = make_dataset_maps(
        maps, valid_nums, labels, train_ratio=0.9, maps_type=maps_type
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        collate_fn=my_collate,
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        collate_fn=my_collate,
    )
    # optimizer
    for param in model_maps.parameters():
        param.requires_grad = False
    for param in model_maps.fc1.parameters():  # only keep feature_map unchanged
        param.requires_grad = True
    for param in model_maps.fc2.parameters():  # feature_map is not feature_all
        param.requires_grad = True
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model_maps.parameters()), lr=0.0001
    )
    # loss funct
    loss_fn = torch.nn.CrossEntropyLoss()

    # acc before refine
    acc = eval_one_epoch(eval_loader, model_maps)
    info = "Deep{} before refine, acc is: {:.3f}".format(maps_type, acc)
    logger.info(info)

    # refine
    model_best = copy.deepcopy(model_maps)
    acc_best = 0.0
    for i in range(epochs):
        epoch_loss = train_one_epoch(train_loader, model_maps, optimizer, loss_fn)
        acc = eval_one_epoch(eval_loader, model_maps)
        info = "Deep{} refine epoch {}, loss: {:.3f}, acc: {:.3f}".format(
            maps_type, i, epoch_loss, acc
        )

        # early stop and save best
        if acc >= acc_best:
            acc_best = acc
            model_best = copy.deepcopy(model_maps)
            patience_counter = 0
            info_best = info
        else:
            patience_counter += 1
        if patience_counter >= cfg.patient:
            info = info_best
            break
    logger.info(info)
    return model_best


def train_model_mall(
    malls: np.ndarray, valid_num: np.ndarray, labels: np.ndarray, epochs: int
) -> torch.nn.Module:
    """
    Train the model DeepMall from scratch on the training set and return the model with optimal performance.

    Parameters
    ----------
    malls : np.ndarray
        The mall data.

    valid_num : np.ndarray
        Valid ion num of each train sample.

    labels : np.ndarray
        The labels of train samples.

    epochs : int
        Number of maximum epochs.

    Returns
    -------
    model_best : torch.nn.Module
        The model with optimal performance.
    """
    batch_size = 64
    num_workers = 0
    train_dataset, eval_dataset_train, mall_dim = make_dataset_mall(
        malls, valid_num, labels
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        collate_fn=my_collate,
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset_train,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        collate_fn=my_collate,
    )

    # model
    model = models.DeepMall(input_dim=mall_dim, feature_dim=32).to(cfg.gpu_id)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    # loss
    loss_fn = torch.nn.CrossEntropyLoss()

    model_best = copy.deepcopy(model)
    acc_best = 0.0
    for epoch in range(epochs):
        epoch_loss = train_one_epoch(train_loader, model, optimizer, loss_fn)
        acc = eval_one_epoch(eval_loader, model)
        # info = 'DeepMall train epoch: {}, loss: {:.3f}, acc: {:.3f}'.format(
        #     epoch, epoch_loss, acc
        # )
        # logger.info(info)

        # early stop and save best
        if acc >= acc_best:
            acc_best = acc
            model_best = copy.deepcopy(model)
            patient_counter = 0
            info = "DeepMall train epoch: {}, loss: {:.3f}, acc: {:.3f}".format(
                epoch, epoch_loss, acc
            )
        else:
            patient_counter += 1
        if patient_counter >= cfg.patient:
            logger.info(info)
            break

    return model_best


def refine_models(
    df_top: pd.DataFrame,
    ms: tims.Tims,
    model_center: torch.nn.Module,
    model_big: torch.nn.Module,
) -> tuple:
    """
    Refine/Train models using the first round identification result.

    Parameters
    ----------
    df_top : pd.DataFrame
        Provide the identification result of peptides.

    ms : tims.Tims
        MS data.

    model_center : torch.nn.Module
        DeepProfile-14 for 14 monoisotope ions.

    model_big : torch.nn.Module
        DeepProfile-56 for monoisotope + isotope ions.

    Returns
    -------
        The fine-tuned model_center, model_big and the trained model_mall.
    """
    logger.info("Extracting maps and malls to refine models...")
    maps_center, maps_big, malls, valid_nums, labels = construct_train_data(df_top, ms)
    # logger.info('Refine models: end to extract maps and malls.')

    model_center = retrain_model_map(
        model_center, maps_center, valid_nums, labels, maps_type="Profile-14", epochs=51
    )
    model_big = retrain_model_map(
        model_big, maps_big, 4 * valid_nums, labels, maps_type="Profile-56", epochs=51
    )
    model_mall = train_model_mall(malls, valid_nums - 3, labels, epochs=51)

    model_center.eval()
    model_big.eval()
    model_mall.eval()

    return model_center, model_big, model_mall
