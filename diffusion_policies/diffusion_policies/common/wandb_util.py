from datetime import datetime
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf


WANDB_PROJECT = "UMI-Gen"
WANDB_DISABLED_MODES = {"disabled", "disable", "off", "false", "none"}


def get_wandb_mode(cfg: DictConfig) -> str:
    mode = OmegaConf.select(cfg, "logging.mode", default="online")
    return str(mode).lower()


def is_wandb_enabled(cfg: DictConfig) -> bool:
    return get_wandb_mode(cfg) not in WANDB_DISABLED_MODES


def infer_policy_prefix(cfg: DictConfig) -> str:
    explicit_prefix = _select_str(cfg, "policy_family")
    if explicit_prefix:
        return explicit_prefix

    target = _select_str(cfg, "policy._target_") or ""
    target_lower = target.lower()
    if "idp3" in target_lower or "improveddp3" in target_lower:
        return "iDP3"
    if "dp3" in target_lower or "pointcloud" in target_lower:
        return "DP3"
    return "DP"


def get_running_setting(cfg: DictConfig) -> str:
    explicit_setting = _select_str(cfg, "logging.running_setting")
    if explicit_setting:
        return explicit_setting

    exp_name = _select_str(cfg, "exp_name")
    if exp_name and exp_name.lower() not in {"default", "debug", "none", "null"}:
        return exp_name

    task_name = _select_str(cfg, "task_name") or _select_str(cfg, "task.name")
    seed = OmegaConf.select(cfg, "training.seed", default=None)
    if task_name and seed is not None:
        return f"{task_name}-seed{seed}"
    if task_name:
        return task_name
    return "default"


def build_wandb_name(cfg: DictConfig, now: Optional[datetime] = None) -> str:
    now = now or datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M%S")
    return f"{infer_policy_prefix(cfg)}-{get_running_setting(cfg)}-{timestamp}"


def build_wandb_kwargs(cfg: DictConfig, output_dir: Optional[str] = None) -> Dict[str, Any]:
    logging_cfg = OmegaConf.select(cfg, "logging", default={})
    kwargs = OmegaConf.to_container(logging_cfg, resolve=True)
    if kwargs is None:
        kwargs = {}

    kwargs["project"] = WANDB_PROJECT
    kwargs["name"] = build_wandb_name(cfg)
    if output_dir is not None:
        kwargs["dir"] = str(output_dir)

    # These are DemoGen-side helpers, not wandb.init keyword arguments.
    kwargs.pop("policy_prefix", None)
    kwargs.pop("running_setting", None)
    return kwargs


def _select_str(cfg: DictConfig, key: str) -> Optional[str]:
    value = OmegaConf.select(cfg, key, default=None)
    if value is None:
        return None
    value = str(value)
    if value == "":
        return None
    return value
