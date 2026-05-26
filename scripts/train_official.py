#!/usr/bin/env python
"""Compatibility launcher for the official omniSAM method ladder.

This script restores the historical ``train_official.py`` entry point used by
the paper export commands. It dispatches to the native stage trainers and
writes a canonical ``checkpoint.pt`` alongside the stage-specific artifact:

- stage 1 variants: ``stage1.pt`` and ``checkpoint.pt``
- staged variants: ``stage2.pt`` and ``checkpoint.pt``

The canonical checkpoint is a native omniSAM checkpoint (``model_state_dict``),
not a source-repo ``head_state_dict`` archive.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


_STAGE1_VARIANTS = {
    "a0_fpn2",
    "a2_fpn2_fpn1_refine",
    "a3_memory",
    "a5_all_at_once",
}
_STAGE2_VARIANTS = {"final_staged", "freeze_a3_control"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _copy_checkpoint(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    shutil.copy2(src, dst)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Official omniSAM method launcher.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--method", default=None)
    parser.add_argument("--init-from-checkpoint", default=None)
    parser.add_argument("--stage1-checkpoint", default=None)
    parser.add_argument("--freeze-stage1-in-stage2", action="store_true")
    parser.add_argument("--epochs-stage1", type=int, default=None)
    parser.add_argument("--epochs-stage2", type=int, default=None)
    parser.add_argument("--allow-nonofficial", action="store_true")
    args = parser.parse_args(argv)

    repo = _repo_root()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.variant in _STAGE1_VARIANTS:
        cmd = [
            sys.executable,
            str(repo / "scripts" / "train_stage1.py"),
            "--config", args.config,
            "--variant", args.variant,
            "--seed", str(args.seed),
            "--output-dir", str(out_dir),
            "--device", args.device,
        ]
        if args.method:
            cmd.extend(["--method", args.method])
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        if args.epochs_stage1 is not None:
            cmd.extend(["--epochs-stage1", str(args.epochs_stage1)])
        if args.allow_nonofficial:
            cmd.append("--allow-nonofficial")
        _run(cmd)
        _copy_checkpoint(out_dir / "stage1.pt", out_dir / "checkpoint.pt")
        return 0

    if args.variant in _STAGE2_VARIANTS:
        stage1_checkpoint = args.stage1_checkpoint or args.init_from_checkpoint
        if not stage1_checkpoint:
            raise SystemExit(
                f"{args.variant} requires --init-from-checkpoint (or --stage1-checkpoint)"
            )
        if args.method:
            method = args.method
        elif args.variant == "freeze_a3_control":
            method = str(repo / "configs" / "official" / "method_freeze_a3_control.yaml")
        else:
            method = str(repo / "configs" / "official" / "method_staged_multiscale.yaml")
        cmd = [
            sys.executable,
            str(repo / "scripts" / "train_stage2.py"),
            "--config", args.config,
            "--method", method,
            "--variant", args.variant,
            "--seed", str(args.seed),
            "--output-dir", str(out_dir),
            "--init-from-checkpoint", str(stage1_checkpoint),
            "--device", args.device,
        ]
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        if args.epochs_stage2 is not None:
            cmd.extend(["--epochs-stage2", str(args.epochs_stage2)])
        if args.freeze_stage1_in_stage2:
            cmd.append("--freeze-stage1-in-stage2")
        if args.allow_nonofficial:
            cmd.append("--allow-nonofficial")
        _run(cmd)
        _copy_checkpoint(out_dir / "stage2.pt", out_dir / "checkpoint.pt")
        return 0

    raise SystemExit(
        f"Unknown variant '{args.variant}'. Known stage1 variants: {sorted(_STAGE1_VARIANTS)}; "
        f"stage2 variants: {sorted(_STAGE2_VARIANTS)}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
