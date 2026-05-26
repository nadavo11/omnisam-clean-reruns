"""Provenance regression tests for MoNuSeg final_staged."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

CANONICAL_METRIC = Path("results/official_metrics/monuseg/final_staged_official_metrics.json")
STALE_DIAGNOSTIC_METRIC = Path("results/official_metrics/monuseg/final_staged/official_metrics.json")
LOCK_PATH = Path("results/official_result_lock.yaml")
CANONICAL_CHECKPOINT = Path(
    "/storage/nada/outputs/fixed_resize_enhancement/monuseg_A5lrn_staged_from_A3s0_e40/checkpoint.pt"
)


def test_monuseg_final_staged_canonical_provenance_schema():
    payload = json.loads(CANONICAL_METRIC.read_text())

    assert payload["dataset"] == "monuseg"
    assert payload["variant"] == "final_staged"
    assert payload["checkpoint"] == str(CANONICAL_CHECKPOINT)
    assert "monuseg_A5lrn_staged_from_A3s0_e40" in payload["checkpoint"]
    assert "keymap" not in payload["checkpoint"]
    assert payload["headline_metrics"]["direct_foreground_dice"] == 0.8539
    assert payload["headline_metrics"]["direct_foreground_iou"] == 0.7459
    assert payload["provenance_status"] == "canonical"

    required_fields = {
        "checkpoint",
        "checkpoint_sha256",
        "config_path",
        "method_config_path",
        "evaluator_script",
        "evaluator_command",
        "created_at",
        "metric_schema_version",
        "provenance_status",
    }
    assert required_fields.issubset(payload.keys())

    lock = yaml.safe_load(LOCK_PATH.read_text())
    lock_entry = lock["monuseg"]["variants"]["final_staged"]
    assert lock_entry["checkpoint_path"] == str(CANONICAL_CHECKPOINT)

    if payload["checkpoint_sha256"] is not None:
        assert lock_entry.get("checkpoint_sha256") == payload["checkpoint_sha256"]

    assert (
        round(payload["headline_metrics"]["direct_foreground_dice"], 4),
        round(payload["headline_metrics"]["direct_foreground_iou"], 4),
    ) != (0.6970, 0.5413)


def test_monuseg_final_staged_diagnostic_copy_is_labeled_noncanonical():
    payload = json.loads(STALE_DIAGNOSTIC_METRIC.read_text())

    assert payload["provenance_status"] == "noncanonical_diagnostic"
    assert payload["variant"] == "final_staged"
