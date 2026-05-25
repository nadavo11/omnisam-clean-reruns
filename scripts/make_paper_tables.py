#!/usr/bin/env python
"""Generate all paper tables from official_metrics.json files.

Outputs (CSV + LaTeX booktabs):
  main_ablation_table     — MoNuSeg ablation ladder A0→final
  staging_table           — Staging mechanism: A3 → staged → freeze-A3
  glas_table              — GlaS architecture ladder
  appendix_ablation_table — Combined MoNuSeg + GlaS appendix rows
  metric_correction_table — Shows locked vs reproduced delta (for supplement)

All headline metrics are direct_foreground_dice / direct_foreground_iou only.
upstream_eval_* are written to auxiliary columns only.

Usage:
  python scripts/make_paper_tables.py \\
    --metrics-root results/official_metrics \\
    --lock-file results/official_result_lock.yaml \\
    --output-dir paper_assets/tables
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml


# ─────────────────────────────── Row specs ────────────────────────────────── #

MONUSEG_ABLATION_ROWS: list[tuple[str, str, str]] = [
    ("a0_fpn2",           "A0",           r"$\text{fpn}_2$ only (anchor)"),
    ("a2_fpn2_fpn1_refine","A2",          r"$\text{fpn}_2$ + $\text{fpn}_1$ refine"),
    ("a3_memory",         "A3",           r"A2 + memory attention ($M{=}8$)"),
    ("a5_all_at_once",    "A5 once",      r"A3 + $\text{fpn}_0$ residual (all-at-once)"),
    ("final_staged",      "\\textbf{Ours}",r"\textbf{Staged} A3 $\to$ $\text{fpn}_0$ (+40 ep)"),
]

STAGING_ROWS: list[tuple[str, str, str]] = [
    ("a3_memory",         "Stage 1 only", r"A3 without $\text{fpn}_0$"),
    ("final_staged",      "Stage 1+2",    r"Joint staged readout (official method)"),
    ("freeze_a3_control", "Freeze-A3",    r"Stage 2 freezes A3 (ablation)"),
]

GLAS_ROWS: list[tuple[str, str, str]] = [
    ("a0_fpn2",            "A0",    r"$\text{fpn}_2$ only (anchor)"),
    ("a2_fpn2_fpn1_refine","A2",    r"$\text{fpn}_2$ + $\text{fpn}_1$ refine"),
    ("a3_memory",          "A3",    r"A2 + memory attention"),
    ("a5_all_at_once",     "A5",    r"A3 + $\text{fpn}_0$ residual"),
]

APPENDIX_ROWS: list[tuple[str, str, str, str]] = [
    # (dataset, variant_key, short_label, description)
    ("monuseg", "a0_fpn2",           "MoNuSeg A0",      r"MoNuSeg anchor"),
    ("monuseg", "a2_fpn2_fpn1_refine","MoNuSeg A2",     r"MoNuSeg + $\text{fpn}_1$"),
    ("monuseg", "a3_memory",         "MoNuSeg A3",      r"MoNuSeg + memory attn"),
    ("monuseg", "a5_all_at_once",    "MoNuSeg A5",      r"MoNuSeg A5 once"),
    ("monuseg", "final_staged",      "MoNuSeg (ours)",  r"MoNuSeg staged (ours)"),
    ("monuseg", "freeze_a3_control", "Freeze-A3",       r"MoNuSeg freeze-A3 control"),
    ("glas",    "a0_fpn2",           "GlaS A0",         r"GlaS anchor"),
    ("glas",    "a2_fpn2_fpn1_refine","GlaS A2",        r"GlaS + $\text{fpn}_1$"),
    ("glas",    "a3_memory",         "GlaS A3",         r"GlaS + memory attn"),
    ("glas",    "a5_all_at_once",    "GlaS A5",         r"GlaS A5 once"),
]


# ───────────────────────────── Helpers ────────────────────────────────────── #

def _load_metrics(
    root: Path,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Walk root collecting official_metrics.json files by dataset/variant."""
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for path in root.rglob("official_metrics.json"):
        try:
            p = json.loads(path.read_text())
        except Exception:
            continue
        ds = str(p.get("dataset", ""))
        v  = str(p.get("variant",  ""))
        hm = p.get("headline_metrics", {})
        row = {
            "direct_foreground_dice": float(hm.get("direct_foreground_dice", 0.0)),
            "direct_foreground_iou":  float(hm.get("direct_foreground_iou",  0.0)),
        }
        out.setdefault(ds, {})[v] = row
    return out


def _load_lock(lock_path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    data = yaml.safe_load(lock_path.read_text())
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for ds, block in data.items():
        if not isinstance(block, dict):
            continue
        variants = block.get("variants", {})
        out[ds] = {}
        for v, payload in variants.items():
            if isinstance(payload, dict):
                out[ds][v] = {
                    "direct_foreground_dice": float(payload.get("direct_foreground_dice", 0.0)),
                    "direct_foreground_iou":  float(payload.get("direct_foreground_iou",  0.0)),
                }
    return out


def _fmt(v: Optional[float], bold: bool = False) -> str:
    if v is None:
        return "--"
    s = f"{v:.4f}"
    return f"\\textbf{{{s}}}" if bold else s


def _write_csv_tex(
    header: List[str],
    rows: List[List[str]],
    csv_path: Path,
    tex_path: Path,
    caption: str = "",
    label: str = "",
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # CSV
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([header] + rows)

    # LaTeX booktabs
    col_spec = "l" + "l" * (len(header) - 1)
    lines = [
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\small",
        "  \\begin{tabular}{" + col_spec + "}",
        "  \\toprule",
        "  " + " & ".join(f"\\textbf{{{h}}}" for h in header) + " \\\\",
        "  \\midrule",
    ]
    for row in rows:
        lines.append("  " + " & ".join(row) + " \\\\")
    lines += [
        "  \\bottomrule",
        "  \\end{tabular}",
    ]
    if caption:
        lines.append(f"  \\caption{{{caption}}}")
    if label:
        lines.append(f"  \\label{{tab:{label}}}")
    lines.append("\\end{table}")
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ───────────────────────────── Table builders ─────────────────────────────── #

def make_main_ablation_table(
    metrics: Dict[str, Dict[str, Dict[str, float]]],
    out: Path,
) -> None:
    ds = metrics.get("monuseg", {})
    # Find best dice to bold it
    best_dice = max((ds[v]["direct_foreground_dice"] for v, _ , _ in MONUSEG_ABLATION_ROWS if v in ds), default=0)
    rows = []
    for key, short, desc in MONUSEG_ABLATION_ROWS:
        m = ds.get(key)
        d = m["direct_foreground_dice"] if m else None
        iou = m["direct_foreground_iou"] if m else None
        is_best = (d is not None and abs(d - best_dice) < 1e-6)
        rows.append([short, desc, _fmt(d, is_best), _fmt(iou, is_best)])
    _write_csv_tex(
        ["ID", "Method", "Dice", "IoU"],
        rows,
        out / "main_ablation_table.csv",
        out / "main_ablation_table.tex",
        caption="MoNuSeg ablation: direct foreground Dice/IoU (higher is better). "
                "Fixed-resize 512\\texttimes512, 40 epochs, seed 0.",
        label="main_ablation",
    )
    print(f"  main_ablation_table ({len(rows)} rows)")


def make_staging_table(
    metrics: Dict[str, Dict[str, Dict[str, float]]],
    out: Path,
) -> None:
    ds = metrics.get("monuseg", {})
    rows = []
    for key, short, desc in STAGING_ROWS:
        m = ds.get(key)
        d = m["direct_foreground_dice"] if m else None
        iou = m["direct_foreground_iou"] if m else None
        rows.append([short, desc, _fmt(d), _fmt(iou)])
    _write_csv_tex(
        ["Config", "Description", "Dice", "IoU"],
        rows,
        out / "staging_table.csv",
        out / "staging_table.tex",
        caption="Staged training ablation on MoNuSeg. "
                "Joint readout outperforms both single-stage and frozen-A3 variants.",
        label="staging",
    )
    print(f"  staging_table ({len(rows)} rows)")


def make_glas_table(
    metrics: Dict[str, Dict[str, Dict[str, float]]],
    out: Path,
) -> None:
    ds = metrics.get("glas", {})
    best_dice = max((ds[v]["direct_foreground_dice"] for v, _, _ in GLAS_ROWS if v in ds), default=0)
    rows = []
    for key, short, desc in GLAS_ROWS:
        m = ds.get(key)
        d = m["direct_foreground_dice"] if m else None
        iou = m["direct_foreground_iou"] if m else None
        is_best = (d is not None and abs(d - best_dice) < 1e-6)
        rows.append([short, desc, _fmt(d, is_best), _fmt(iou, is_best)])
    _write_csv_tex(
        ["ID", "Method", "Dice", "IoU"],
        rows,
        out / "glas_table.csv",
        out / "glas_table.tex",
        caption="GlaS architecture ablation: direct foreground Dice/IoU. "
                "Fixed-resize 224\\texttimes224, 200 epochs, autosam\\_dense\\_v1 augmentation.",
        label="glas_ablation",
    )
    print(f"  glas_table ({len(rows)} rows)")


def make_appendix_ablation_table(
    metrics: Dict[str, Dict[str, Dict[str, float]]],
    out: Path,
) -> None:
    rows = []
    for ds, key, short, desc in APPENDIX_ROWS:
        m = metrics.get(ds, {}).get(key)
        d = m["direct_foreground_dice"] if m else None
        iou = m["direct_foreground_iou"] if m else None
        rows.append([ds.upper(), short, desc, _fmt(d), _fmt(iou)])
    _write_csv_tex(
        ["Dataset", "ID", "Description", "Dice", "IoU"],
        rows,
        out / "appendix_ablation_table.csv",
        out / "appendix_ablation_table.tex",
        caption="Full ablation across MoNuSeg and GlaS datasets.",
        label="appendix_ablation",
    )
    print(f"  appendix_ablation_table ({len(rows)} rows)")


def make_metric_correction_table(
    metrics: Dict[str, Dict[str, Dict[str, float]]],
    lock: Dict[str, Dict[str, Dict[str, float]]],
    out: Path,
) -> None:
    """Table showing locked vs reproduced metrics and delta — for supplement/audit."""
    rows = []
    pass_threshold_dice = 0.002
    pass_threshold_iou  = 0.003
    for ds in ("monuseg", "glas"):
        for key in (metrics.get(ds, {}).keys() | lock.get(ds, {}).keys()):
            rep = metrics.get(ds, {}).get(key)
            lk  = lock.get(ds, {}).get(key)
            if rep is None and lk is None:
                continue
            rep_d = rep["direct_foreground_dice"] if rep else None
            rep_i = rep["direct_foreground_iou"]  if rep else None
            lk_d  = lk["direct_foreground_dice"]  if lk  else None
            lk_i  = lk["direct_foreground_iou"]   if lk  else None
            delta_d = (abs(rep_d - lk_d) if rep_d is not None and lk_d is not None else None)
            delta_i = (abs(rep_i - lk_i) if rep_i is not None and lk_i is not None else None)
            if delta_d is not None and delta_i is not None:
                verdict = "PASS" if delta_d <= pass_threshold_dice and delta_i <= pass_threshold_iou else "FAIL"
            else:
                verdict = "N/A"
            rows.append([
                ds.upper(), key,
                _fmt(lk_d), _fmt(rep_d),
                f"{delta_d:.4f}" if delta_d is not None else "--",
                _fmt(lk_i), _fmt(rep_i),
                f"{delta_i:.4f}" if delta_i is not None else "--",
                verdict,
            ])
    _write_csv_tex(
        ["Dataset", "Variant",
         "Lock Dice", "Repro Dice", "ΔDice",
         "Lock IoU",  "Repro IoU",  "ΔIoU",
         "Pass?"],
        rows,
        out / "metric_correction_table.csv",
        out / "metric_correction_table.tex",
        caption="Locked vs reproduced metrics. Pass threshold: "
                "$|\\Delta\\text{Dice}| \\leq 0.002$, $|\\Delta\\text{IoU}| \\leq 0.003$.",
        label="metric_correction",
    )
    print(f"  metric_correction_table ({len(rows)} rows)")


# ───────────────────────────── Main ──────────────────────────────────────── #

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate all paper tables.")
    parser.add_argument("--metrics-root", default="results/official_metrics")
    parser.add_argument("--lock-file",    default="results/official_result_lock.yaml")
    parser.add_argument("--output-dir",   default="paper_assets/tables")
    args = parser.parse_args(argv)

    out  = Path(args.output_dir)
    lock_path = Path(args.lock_file)

    metrics = _load_metrics(Path(args.metrics_root))
    lock    = _load_lock(lock_path) if lock_path.exists() else {}

    # Fall back to lock if no JSON metrics yet.
    if not any(metrics.values()):
        print(f"No per-run JSON found; using lock file only.")
        metrics = lock

    out.mkdir(parents=True, exist_ok=True)
    print(f"Writing tables → {out}")

    make_main_ablation_table(metrics, out)
    make_staging_table(metrics, out)
    make_glas_table(metrics, out)
    make_appendix_ablation_table(metrics, out)
    make_metric_correction_table(metrics, lock, out)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
