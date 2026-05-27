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
    # Batch 1 — completed 2026-05-26
    "A3M4": "configs/test_screen/monuseg_a3_m4.yaml",
    "A3M8": "configs/test_screen/monuseg_a3_m8.yaml",
    "A5staged": "configs/test_screen/monuseg_a5_staged_from_a3.yaml",
    "A5noMem": "configs/test_screen/monuseg_a5_staged_no_memory.yaml",
    # Batch 2 — Priority A: fixed short schedule controls
    "A2short20": "configs/test_screen/monuseg_a2_short20.yaml",
    "A3M4short20": "configs/test_screen/monuseg_a3m4_short20.yaml",
    "A3M8short20": "configs/test_screen/monuseg_a3m8_short20.yaml",
    # Batch 2 — Priority B: cosine schedule controls
    "A2cos40": "configs/test_screen/monuseg_a2_cosine40.yaml",
    "A3M4cos40": "configs/test_screen/monuseg_a3m4_cosine40.yaml",
    "A3M8cos40": "configs/test_screen/monuseg_a3m8_cosine40.yaml",
    # Batch 2 — Priority C: corrected staged schedule (stage1=20)
    "A5stage20fromA3M4": "configs/test_screen/monuseg_a5_staged20_from_a3m4.yaml",
    "A5stage20noMem": "configs/test_screen/monuseg_a5_staged20_no_memory.yaml",
    # Batch 3 — augmentation variants (autosam_dense_v1)
    "A3M4short20aug": "configs/test_screen/monuseg_a3m4_short20_aug.yaml",
    "A3M4cos40aug": "configs/test_screen/monuseg_a3m4_cosine40_aug.yaml",
    "A5stage20noMemAug": "configs/test_screen/monuseg_a5_staged20_no_memory_aug.yaml",
}

VARIANT_NAMES = {
    "A3M4": "monuseg-testscreen-a3m4",
    "A3M8": "monuseg-testscreen-a3m8",
    "A5staged": "monuseg-testscreen-a5staged-from-a3",
    "A5noMem": "monuseg-testscreen-a5staged-no-memory",
    "A2short20": "monuseg-testscreen-a2-short20",
    "A3M4short20": "monuseg-testscreen-a3m4-short20",
    "A3M8short20": "monuseg-testscreen-a3m8-short20",
    "A2cos40": "monuseg-testscreen-a2-cos40",
    "A3M4cos40": "monuseg-testscreen-a3m4-cos40",
    "A3M8cos40": "monuseg-testscreen-a3m8-cos40",
    "A5stage20fromA3M4": "monuseg-testscreen-a5stage20-a3m4",
    "A5stage20noMem": "monuseg-testscreen-a5stage20-nomem",
    "A3M4short20aug": "monuseg-testscreen-a3m4-short20-aug",
    "A3M4cos40aug": "monuseg-testscreen-a3m4-cos40-aug",
    "A5stage20noMemAug": "monuseg-testscreen-a5stage20-nomem-aug",
}


def pvc_target(variant_key: str, seed: int) -> str:
    return f"/storage/nada/test_screen/{variant_key.lower()}/s{seed}"


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
        "WANDB_PROJECT=frozen-sam-readout",
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
        default="A3M4short20aug,A3M4cos40aug,A5stage20noMemAug",
        help="Comma-separated variant keys (default: batch 3 augmentation variants)",
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
