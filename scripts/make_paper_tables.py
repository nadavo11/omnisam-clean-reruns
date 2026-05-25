#!/usr/bin/env python
"""Read ``official_metrics.json`` files and produce paper tables.

Outputs:
  paper_assets/tables/main_ablation_table.csv  + .tex
  paper_assets/tables/staging_table.csv        + .tex

If individual JSON metric files are not yet produced, the script falls back to
``results/official_result_lock.yaml`` so the tables can still be rendered.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import yaml


VARIANTS_ROW_ORDER = [
    ("a0_fpn2", "A0 (fpn_2 only)"),
    ("a2_fpn2_fpn1_refine", "A2 (fpn_2 + fpn_1 refine)"),
    ("a3_memory", "A3 (A2 + memory attn)"),
    ("a5_all_at_once", "A5 all-at-once"),
    ("final_staged", "Final staged (official)"),
    ("freeze_a3_control", "Freeze-A3 control"),
]


def _collect_from_json(metrics_root: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Walk metrics_root looking for ``official_metrics.json`` files."""

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for path in metrics_root.rglob("official_metrics.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        dataset = str(payload.get("dataset", "unknown"))
        variant = str(payload.get("variant", "unknown"))
        head = payload.get("headline_metrics", {})
        out.setdefault(dataset, {})[variant] = {
            "direct_foreground_dice": float(head.get("direct_foreground_dice", 0.0)),
            "direct_foreground_iou": float(head.get("direct_foreground_iou", 0.0)),
        }
    return out


def _collect_from_lock(lock_path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    data = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for dataset_name, dataset_block in data.items():
        variants = dataset_block.get("variants", {})
        out[dataset_name] = {
            v: {
                "direct_foreground_dice": float(payload.get("direct_foreground_dice", 0.0)),
                "direct_foreground_iou": float(payload.get("direct_foreground_iou", 0.0)),
            }
            for v, payload in variants.items()
        }
    return out


def _write_table(rows: List[List[str]], csv_path: Path, tex_path: Path, header: List[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    tex_lines = [
        "\\begin{tabular}{" + "l" * len(header) + "}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        tex_lines.append(" & ".join(row) + " \\\\")
    tex_lines.append("\\bottomrule")
    tex_lines.append("\\end{tabular}")
    tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-root", default="results/official_metrics")
    parser.add_argument("--lock-file", default="results/official_result_lock.yaml")
    parser.add_argument("--output-dir", default="paper_assets/tables")
    args = parser.parse_args(argv)

    metrics_root = Path(args.metrics_root)
    lock_path = Path(args.lock_file)
    out_dir = Path(args.output_dir)

    collected = _collect_from_json(metrics_root)
    if not collected and lock_path.is_file():
        print(f"No per-run JSON found; using lock file {lock_path}.")
        collected = _collect_from_lock(lock_path)

    # Main ablation: MoNuSeg rows.
    main_rows: List[List[str]] = []
    monuseg_metrics = collected.get("monuseg", {})
    for key, description in VARIANTS_ROW_ORDER:
        metrics = monuseg_metrics.get(key)
        if metrics is None:
            continue
        main_rows.append([
            key,
            description,
            f"{metrics['direct_foreground_dice']:.4f}",
            f"{metrics['direct_foreground_iou']:.4f}",
        ])
    _write_table(
        main_rows,
        csv_path=out_dir / "main_ablation_table.csv",
        tex_path=out_dir / "main_ablation_table.tex",
        header=["Variant", "Description", "Dice", "IoU"],
    )

    # Staging table: A3 -> Final staged -> Freeze-A3 control.
    staging_keys = ["a3_memory", "final_staged", "freeze_a3_control"]
    staging_rows: List[List[str]] = []
    for key in staging_keys:
        metrics = monuseg_metrics.get(key)
        if metrics is None:
            continue
        desc = dict(VARIANTS_ROW_ORDER)[key]
        staging_rows.append([
            key,
            desc,
            f"{metrics['direct_foreground_dice']:.4f}",
            f"{metrics['direct_foreground_iou']:.4f}",
        ])
    _write_table(
        staging_rows,
        csv_path=out_dir / "staging_table.csv",
        tex_path=out_dir / "staging_table.tex",
        header=["Variant", "Description", "Dice", "IoU"],
    )

    print(f"Wrote tables under {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
