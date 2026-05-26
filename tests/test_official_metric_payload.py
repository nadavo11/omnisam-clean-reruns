"""Regression tests for the official metrics JSON payload."""

from __future__ import annotations

from frozen_sam_readout.evaluation.prediction_io import build_official_metrics_payload


def test_official_metrics_payload_records_eval_split():
    payload = build_official_metrics_payload(
        dataset="monuseg",
        protocol="monuseg_strict_512",
        variant="final_staged",
        checkpoint="/tmp/checkpoint.pt",
        eval_split="test",
        seed=0,
        dice=0.1,
        iou=0.2,
        threshold=0.5,
        prediction_resolution="72x72",
        gt_resolution="1000x1000",
    )

    assert payload["eval_split"] == "test"
    assert payload["paper_headline_safe"] is True

