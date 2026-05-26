"""Official ``official_metrics.json`` writer."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict


OFFICIAL_METRIC_SOURCE = "frozen_sam_readout/evaluation/metrics.py::compute_binary_metrics"


def build_official_metrics_payload(
    *,
    dataset: str,
    protocol: str,
    variant: str,
    checkpoint: str,
    eval_split: str,
    checkpoint_sha256: str | None = None,
    config_path: str | None = None,
    method_config_path: str | None = None,
    evaluator_script: str | None = None,
    evaluator_command: str | None = None,
    created_at: str | None = None,
    metric_schema_version: int = 2,
    provenance_status: str = "canonical",
    seed: int,
    dice: float,
    iou: float,
    threshold: float,
    prediction_resolution: str,
    gt_resolution: str,
    paper_headline_safe: bool = True,
    upstream_eval_dice: float | None = None,
    upstream_eval_iou: float | None = None,
) -> Dict[str, Any]:
    return {
        "dataset": str(dataset),
        "protocol": str(protocol),
        "variant": str(variant),
        "checkpoint": str(checkpoint),
        "eval_split": str(eval_split),
        "checkpoint_sha256": checkpoint_sha256,
        "config_path": config_path,
        "method_config_path": method_config_path,
        "evaluator_script": evaluator_script,
        "evaluator_command": evaluator_command,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "metric_schema_version": int(metric_schema_version),
        "provenance_status": str(provenance_status),
        "seed": int(seed),
        "headline_metrics": {
            "direct_foreground_dice": float(dice),
            "direct_foreground_iou": float(iou),
        },
        "auxiliary_metrics": {
            "upstream_eval_dice": upstream_eval_dice,
            "upstream_eval_iou": upstream_eval_iou,
        },
        "threshold": float(threshold),
        "metric_source": OFFICIAL_METRIC_SOURCE,
        "prediction_resolution": str(prediction_resolution),
        "gt_resolution": str(gt_resolution),
        "paper_headline_safe": bool(paper_headline_safe),
    }


def write_official_metrics_json(path: str | Path, payload: Dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path
