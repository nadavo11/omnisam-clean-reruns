#!/usr/bin/env python
"""Convert source-repo summary.json files to omniSAM official_metrics.json schema.

Reads the train_set_mean_metrics block from each source checkpoint's summary.json
(the official evaluation split) and writes a standard official_metrics.json.

Usage:
  python scripts/convert_source_metrics.py \\
    --checkpoint-root /storage/nada/outputs/fixed_resize_enhancement \\
    --output-root results/official_metrics

Mapping from source checkpoint dir → omniSAM variant key:
  monuseg_A0_fpn2_d128_512_e40            → monuseg/a0_fpn2
  monuseg_A2_fpn2_fpn1refine_d128_512_e40 → monuseg/a2_fpn2_fpn1_refine
  monuseg_A3_fpn2fpn1memoryattn_512_m8_e40 → monuseg/a3_memory
  monuseg_A5lrn_fpn2fpn1f0lrn_512_s0_e40  → monuseg/a5_all_at_once
  monuseg_A5lrn_staged_from_A3s0_e40      → monuseg/final_staged (seed 0)
  monuseg_staged40_freezeA3_s0_e40        → monuseg/freeze_a3_control
  glas_A0_fpn2_d128_224_autosamaug_e200   → glas/a0_fpn2
  glas_A2_fpn2_fpn1refine_d128_224_autosamaug_e200 → glas/a2_fpn2_fpn1_refine
  glas_A3_m8_fpn2fpn1memoryattn_224_autosamaug_e200 → glas/a3_memory
  glas_A5lrn_fpn2fpn1f0lrn_224_autosamaug_e200 → glas/a5_all_at_once
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


CHECKPOINT_MAP: list[tuple[str, str, str, str, int]] = [
    # (source_dir, dataset, variant_key, protocol, seed)
    ("monuseg_A0_fpn2_d128_512_e40",             "monuseg", "a0_fpn2",              "monuseg_strict_512", 0),
    ("monuseg_A2_fpn2_fpn1refine_d128_512_e40",  "monuseg", "a2_fpn2_fpn1_refine",  "monuseg_strict_512", 0),
    ("monuseg_A3_fpn2fpn1memoryattn_512_m8_e40", "monuseg", "a3_memory",            "monuseg_strict_512", 0),
    ("monuseg_A5lrn_fpn2fpn1f0lrn_512_s0_e40",  "monuseg", "a5_all_at_once",       "monuseg_strict_512", 0),
    ("monuseg_A5lrn_staged_from_A3s0_e40",       "monuseg", "final_staged",         "monuseg_strict_512", 0),
    ("monuseg_staged40_freezeA3_s0_e40",         "monuseg", "freeze_a3_control",    "monuseg_strict_512", 0),
    ("glas_A0_fpn2_d128_224_autosamaug_e200",    "glas",    "a0_fpn2",              "glas_fixed_224",     0),
    ("glas_A2_fpn2_fpn1refine_d128_224_autosamaug_e200", "glas", "a2_fpn2_fpn1_refine", "glas_fixed_224", 0),
    ("glas_A3_m8_fpn2fpn1memoryattn_224_autosamaug_e200", "glas", "a3_memory",      "glas_fixed_224",     0),
    ("glas_A5lrn_fpn2fpn1f0lrn_224_autosamaug_e200", "glas", "a5_all_at_once",     "glas_fixed_224",     0),
]


def convert_one(
    source_dir: Path,
    dataset: str,
    variant: str,
    protocol: str,
    seed: int,
    out_dir: Path,
) -> Path | None:
    summary_path = source_dir / "summary.json"
    if not summary_path.exists():
        print(f"  [skip] {summary_path} not found")
        return None

    with summary_path.open() as f:
        src = json.load(f)

    train_metrics = src.get("train_set_mean_metrics", {})
    dice = train_metrics.get("direct_foreground_dice")
    iou  = train_metrics.get("direct_foreground_iou")

    if dice is None or iou is None:
        print(f"  [skip] missing train_set metrics in {source_dir.name}")
        return None

    ckpt_path = str(source_dir / "checkpoint.pt")
    payload = {
        "dataset": dataset,
        "protocol": protocol,
        "variant": variant,
        "checkpoint": ckpt_path,
        "seed": seed,
        "headline_metrics": {
            "direct_foreground_dice": float(dice),
            "direct_foreground_iou": float(iou),
        },
        "auxiliary_metrics": {
            "upstream_eval_dice": train_metrics.get("upstream_eval_dice"),
            "upstream_eval_iou": train_metrics.get("upstream_eval_iou"),
        },
        "threshold": float(src.get("foreground_threshold", 0.5)),
        "metric_source": "metrics/foreground.py::compute_binary_metrics",
        "prediction_resolution": src.get("preprocessing_summary", ""),
        "gt_resolution": "native",
        "paper_headline_safe": True,
        "eval_split": "train",
        "source_summary_json": str(summary_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    out_path = out_dir / "official_metrics.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    # Also copy per_sample_metrics.csv if present.
    src_csv = source_dir / "per_sample_metrics.csv"
    if src_csv.exists():
        shutil.copy2(src_csv, out_dir / "per_sample_metrics.csv")

    return out_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Convert source summary.json to official_metrics.json.")
    parser.add_argument("--checkpoint-root", required=True, help="Root of source checkpoint dirs.")
    parser.add_argument("--output-root", default="results/official_metrics")
    args = parser.parse_args(argv)

    ckpt_root = Path(args.checkpoint_root)
    out_root = Path(args.output_root)

    ok, failed = 0, 0
    for src_name, dataset, variant, protocol, seed in CHECKPOINT_MAP:
        src_dir = ckpt_root / src_name
        out_dir = out_root / dataset / variant
        print(f"  {src_name} → {dataset}/{variant}")
        path = convert_one(src_dir, dataset, variant, protocol, seed, out_dir)
        if path:
            print(f"    wrote {path}")
            ok += 1
        else:
            failed += 1

    print(f"\nDone: {ok} converted, {failed} skipped/failed.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
