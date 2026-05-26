#!/usr/bin/env python3
"""Submit MoNuSeg TEST_SCREEN ablation jobs through RunAI.

TEST_SCREEN protocol:
- Train on full MoNuSeg training set (no val holdout).
- Evaluate on official test split every epoch.
- Checkpoint selected by best test Dice (transparent test-screening, no cherry-picking).
- All results go into test-screen ledger.

Job names: monuseg-testscreen-{variant}-s{seed}
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


IMAGE = "pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime"
PROJECT = "avidan"
REPO_GIT = "https://github.com/nadavo11/omnisam-clean-launcher.git"

VARIANTS = {
    "A3M4": "configs/test_screen/monuseg_a3_m4.yaml",
    "A3M8": "configs/test_screen/monuseg_a3_m8.yaml",
    "A5staged": "configs/test_screen/monuseg_a5_staged_from_a3.yaml",
    "A5noMem": "configs/test_screen/monuseg_a5_staged_no_memory.yaml",
}

VARIANT_NAMES = {
    "A3M4": "monuseg-testscreen-A3M4",
    "A3M8": "monuseg-testscreen-A3M8",
    "A5staged": "monuseg-testscreen-A5staged-from-A3",
    "A5noMem": "monuseg-testscreen-A5staged-no-memory",
}


def pvc_target(variant_key: str, seed: int) -> str:
    return f"/storage/nada/test_screen/{variant_key}/s{seed}"


def submit_job(
    variant_key: str,
    config_path: str,
    seed: int,
    *,
    dry_run: bool,
) -> None:
    base_name = VARIANT_NAMES[variant_key]
    job_name = f"{base_name}-s{seed}"
    run_name = job_name.replace("-", "_")
    sync_target = pvc_target(variant_key, seed)
    repo_dir = f"{sync_target}/omnisam-clean-launcher.git"

    job_cmd = [
        "python",
        f"{repo_dir}/launch_monuseg_test_screen.py",
        "--config",
        config_path,
        "--seed",
        str(seed),
        "--run-name",
        run_name,
        "--wandb-group",
        "monuseg_test_screen_ablation",
        "--wandb-job-type",
        "TEST_SCREEN",
    ]

    cmd = [
        "runai",
        "submit",
        "--name",
        job_name,
        "--image",
        IMAGE,
        "--gpu",
        "1",
        "--project",
        PROJECT,
        "--cpu",
        "4",
        "--memory",
        "32G",
        "--image-pull-policy",
        "IfNotPresent",
        "--existing-pvc",
        "claimname=storage,path=/storage",
        "--working-dir",
        "/storage/nada",
        "--git-sync",
        f"source={REPO_GIT},branch=main,target={sync_target}",
        "--pod-running-timeout",
        "20m",
        "--environment",
        "RUNAI_STORAGE_ROOT=/storage/nada",
        "--environment",
        "RUNAI_CACHE_ROOT=/storage/nada/cache/texture-representations",
        "--environment",
        "RUNAI_VENV_ROOT=/storage/nada/envs/texture-representations-runai-cu128",
        "--environment",
        "RUNAI_ENABLE_PERSISTENT_VENV=1",
        "--environment",
        "WANDB_MODE=online",
        "--environment",
        f"WANDB_PROJECT=nadavoteam/frozen-sam-readout",
    ]
    for env_name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "WANDB_API_KEY"):
        env_value = os.environ.get(env_name)
        if env_value:
            cmd.extend(["--environment", f"{env_name}={env_value}"])
    cmd.extend(["--command", "--", *job_cmd])

    print(f"\n### {job_name}")
    print(" ".join(subprocess.list2cmdline([part]) for part in cmd))
    if dry_run:
        return
    proc = subprocess.run(cmd, text=True, capture_output=True)
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without submitting")
    parser.add_argument("--seeds", default="0,1,2", help="Comma-separated seeds")
    parser.add_argument(
        "--variants",
        default="A3M4,A3M8,A5staged,A5noMem",
        help="Comma-separated variant keys",
    )
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    for variant in variants:
        if variant not in VARIANTS:
            raise SystemExit(f"Unknown variant '{variant}'. Valid: {list(VARIANTS)}")
        config_path = VARIANTS[variant]
        for seed in seeds:
            submit_job(variant, config_path, seed, dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
