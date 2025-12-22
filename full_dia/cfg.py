from pathlib import Path

import yaml

from full_dia.log import Logger

logger = Logger.get_logger()

params = {}


def flatten_yaml(cfg_dict: dict) -> dict:
    """
    Remove the first domain for a yaml file.
    """
    result = {}
    for k, v in cfg_dict.items():
        if isinstance(v, dict):
            result = result | v
        else:
            result[k] = v
    return result


def load_default():
    """
    Load the default.yaml file in cfg folder
    """
    global params
    default_path = Path(__file__).parent / "cfg" / "default.yaml"
    with open(default_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    params = flatten_yaml(raw)

    for k, v in params.items():
        globals()[k] = v


def update_from_yaml(yaml_path):
    """
    Update params from a yaml file provided by '-cfg_develop' param.
    """
    if yaml_path is None:
        return

    global params

    yaml_path = Path(yaml_path)
    if not (yaml_path.is_file() and yaml_path.suffix.lower() in {".yml", ".yaml"}):
        raise ValueError(f"Invalid YAML config file: {yaml_path}")

    with open(yaml_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    params_new = flatten_yaml(raw)

    # log changed params
    for k, v in params_new.items():
        if k in params and params[k] != v:
            info = "param changed: {}, {} -> {}".format(k, params[k], v)
            logger.info(info)

    params = {**params, **params_new}

    for k, v in params.items():
        globals()[k] = v
