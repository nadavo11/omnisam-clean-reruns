"""YAML config loader and official-protocol guard validators."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigGuardError(RuntimeError):
    """Raised when an official-protocol guard is violated."""


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Top-level YAML at '{path}' must be a mapping.")
    return loaded


def validate_official_guards(config: Dict[str, Any], *, allow_nonofficial: bool = False) -> None:
    """Fail loudly if an official-protocol invariant is violated.

    Guards (per ``guards`` section, defaulting to True):
    - ``fail_if_sam_trainable`` -> ``sam.frozen`` must be ``True``
    - ``fail_if_prompt_encoder_used`` -> ``sam.use_prompt_encoder`` must be ``False``
    - ``fail_if_mask_decoder_used`` -> ``sam.use_mask_decoder`` must be ``False``
    - ``evaluation.split`` must be ``test`` for official evaluation
    - ``evaluation.threshold`` must equal ``0.5``
    """

    if allow_nonofficial:
        return

    sam = config.get("sam", {})
    guards = config.get("guards", {})
    eval_cfg = config.get("evaluation", {})

    if guards.get("fail_if_sam_trainable", True) and not sam.get("frozen", True):
        raise ConfigGuardError("Official guard: SAM must be frozen (sam.frozen=true).")
    if guards.get("fail_if_prompt_encoder_used", True) and sam.get("use_prompt_encoder", False):
        raise ConfigGuardError(
            "Official guard: SAM prompt encoder must not be used (sam.use_prompt_encoder=false)."
        )
    if guards.get("fail_if_mask_decoder_used", True) and sam.get("use_mask_decoder", False):
        raise ConfigGuardError(
            "Official guard: SAM mask decoder must not be used (sam.use_mask_decoder=false)."
        )
    eval_split = str(eval_cfg.get("split", "test"))
    if eval_split != "test":
        raise ConfigGuardError(
            f"Official guard: official evaluation must run on the test split (got evaluation.split={eval_split!r})."
        )
    threshold = eval_cfg.get("threshold", 0.5)
    if float(threshold) != 0.5:
        raise ConfigGuardError(
            f"Official guard: prediction threshold must be 0.5 (got {threshold})."
        )
