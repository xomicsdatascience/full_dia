#!/usr/bin/env python
import argparse
from pathlib import Path

from full_dia import cfg, cross, utils
from full_dia.library import Library
from full_dia.log import Logger
from full_dia.search import search_core

try:
    _ = profile
except NameError:

    def profile(func):
        return func


logger = Logger.get_logger()


def bootstrap(args: argparse.Namespace) -> None:
    """
    Initialize tasks.
    """
    # create out folder
    out_dir = Path(args.ws) / args.out_name
    out_dir.mkdir(exist_ok=True)
    # init log
    Logger.set_logger(out_dir)
    # print info
    utils.check_run_info(args)
    # init cfg
    cfg.load_default()
    cfg.update_from_yaml(args.cfg_develop)
    # init ws
    utils.init_multi_ws(Path(args.ws), args.out_name)
    # init gpu
    utils.init_gpu_params(args.gpu_id)
    # others
    cfg.is_compare_mode = args.compare
    cfg.is_overwrite = args.overwrite
    if args.low_memory:
        cfg.target_batch_max = cfg.target_batch_max / 2.0

    # check
    if cfg.file_num < 2:
        info = "Full-DIA needs >= 2 runs to complete the analysis!"
        logger.warning(info)


@profile
def main():
    # init
    args = utils.get_args()
    bootstrap(args)

    # lib
    lib = Library(args.lib)

    # search
    search_core(lib)

    # global
    logger.info("=================Global Analysis=================")
    df_global = cross.perform_global(
        lib, cfg.top_k_fg, cfg.top_k_pr, multi_ws=cfg.multi_ws
    )
    cross.save_report_result(df_global, multi_ws=cfg.multi_ws)
    logger.info("Finished.")
    return


if __name__ == "__main__":
    main()
