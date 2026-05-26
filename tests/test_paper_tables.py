"""Regression tests for paper table metric selection."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.make_paper_tables import _load_metrics


def _write_metrics(path: Path, dice: float, iou: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "dataset": "monuseg",
                "variant": "final_staged",
                "checkpoint": "/storage/nada/outputs/fixed_resize_enhancement/monuseg_A5lrn_staged_from_A3s0_e40/checkpoint.pt",
                "checkpoint_sha256": None,
                "config_path": "/home/nada/PycharmProjects/omniSAM/configs/official/monuseg_strict_512.yaml",
                "method_config_path": "/home/nada/PycharmProjects/omniSAM/configs/official/method_staged_multiscale.yaml",
                "evaluator_script": "/home/nada/PycharmProjects/omniSAM/scripts/eval_official.py",
                "evaluator_command": "python scripts/eval_official.py ...",
                "created_at": "2026-05-25T20:58:02.474255+00:00",
                "metric_schema_version": 2,
                "provenance_status": "canonical",
                "headline_metrics": {
                    "direct_foreground_dice": dice,
                    "direct_foreground_iou": iou,
                },
            }
        )
    )


def test_load_metrics_prefers_canonical_non_keymap_file(tmp_path):
    canonical = tmp_path / "monuseg" / "final_staged_official_metrics.json"
    stale = tmp_path / "monuseg" / "final_staged" / "official_metrics.json"
    _write_metrics(canonical, 0.8539, 0.7459)
    _write_metrics(stale, 0.6970, 0.5413)
    stale_payload = json.loads(stale.read_text())
    stale_payload["provenance_status"] = "noncanonical_diagnostic"
    stale.write_text(json.dumps(stale_payload))

    lock = {
        "monuseg": {
            "final_staged": {
                "checkpoint_path": "/storage/nada/outputs/fixed_resize_enhancement/monuseg_A5lrn_staged_from_A3s0_e40/checkpoint.pt",
                "direct_foreground_dice": 0.8545,
                "direct_foreground_iou": 0.7467,
            }
        }
    }
    metrics = _load_metrics(tmp_path, lock=lock)
    row = metrics["monuseg"]["final_staged"]
    assert row["direct_foreground_dice"] == 0.8539
    assert row["direct_foreground_iou"] == 0.7459
