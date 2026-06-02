#!/usr/bin/env python3
"""Monitor the final MoNuSeg capacity sweep and write a live ledger."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import time
from pathlib import Path
from typing import Any


LEDGER_PATH = Path("reports/monuseg_final_capacity160_live_ledger.md")


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if v != v:
            return "nan"
        return f"{v:.6f}"
    return str(v)


def _collect_runs(group: str):
    import wandb

    api = wandb.Api(timeout=120)
    path = f"{os.environ.get('WANDB_ENTITY', 'nadavoteam')}/frozen-sam-readout"
    runs = api.runs(path, filters={"group": group}, per_page=500, lazy=True)
    return list(runs)


def _write_ledger(group: str, runs: list[Any]) -> None:
    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [
        "# MonuSeg Final Capacity 160 Live Ledger",
        "",
        f"- Group: `{group}`",
        f"- Updated: `{now}`",
        f"- Runs seen: `{len(runs)}`",
        "",
        "| run | state | epoch | ep80 Dice | ep80 IoU | ep120 Dice | ep120 IoU | final Dice | final IoU | best Dice | best IoU | best epoch | alpha | W&B URL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for run in sorted(runs, key=lambda r: (r.name or "", r.state or "")):
        summary = run.summary or {}
        epoch = summary.get("epoch", summary.get("_step", ""))
        url = getattr(run, "url", "")
        best_epoch = summary.get("best_test_epoch", summary.get("checkpoint_views/best_test_epoch/selected_epoch", ""))
        alpha = summary.get("model/alpha", summary.get("attention/residual/alpha", None))
        lines.append(
            "| "
            + " | ".join([
                run.name,
                run.state,
                _fmt(epoch),
                _fmt(summary.get("test/epoch80_dice", None)),
                _fmt(summary.get("test/epoch80_iou", None)),
                _fmt(summary.get("test/epoch120_dice", None)),
                _fmt(summary.get("test/epoch120_iou", None)),
                _fmt(summary.get("test/final_dice", summary.get("test/dice", None))),
                _fmt(summary.get("test/final_iou", summary.get("test/iou", None))),
                _fmt(summary.get("test/dice", None)),
                _fmt(summary.get("test/iou", None)),
                _fmt(best_epoch),
                _fmt(alpha),
                url,
            ])
            + " |"
        )
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEDGER_PATH.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", default="monuseg_d0fpn_final_capacity160")
    parser.add_argument("--interval", type=int, default=600, help="Polling interval in seconds")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    while True:
        try:
            runs = _collect_runs(args.group)
            _write_ledger(args.group, runs)
            print(f"[info] wrote {LEDGER_PATH} with {len(runs)} runs")
        except Exception as exc:
            print(f"[warn] monitoring failed: {exc}")
        if args.once:
            return 0
        time.sleep(max(60, int(args.interval)))


if __name__ == "__main__":
    raise SystemExit(main())
