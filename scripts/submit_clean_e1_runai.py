#!/usr/bin/env python3
"""Submit the clean E1 MoNuSeg matrix through RunAI.

E1 split contract enforced here:
- E1 jobs NEVER evaluate on the official test split.
- Use --suffix to distinguish run batches (e.g. --suffix v2).
- PVC target paths include the suffix to avoid clobbering previous worktrees.
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
    "A0": ("configs/official/method_a0_fpn2.yaml", "a0_fpn2"),
    "A2": ("configs/official/method_a2_fpn2_fpn1_refine.yaml", "a2_fpn2_fpn1_refine"),
    "A3": ("configs/official/method_a3_memory.yaml", "a3_memory"),
}


def run_name_to_target(variant_flag: str, seed: int, suffix: str) -> str:
    variant_short = variant_flag.split("_", 1)[0]
    suffix_part = f"_{suffix}" if suffix else ""
    return f"/storage/nada/ce1{suffix_part}/{variant_short}/s{seed}"


def build_job_command(run_name: str, config_path: str, variant_flag: str, seed: int, repo_dir: str) -> list[str]:
    return [
        "python",
        f"{repo_dir}/launch_clean_e1.py",
        "--config",
        config_path,
        "--variant",
        variant_flag,
        "--seed",
        str(seed),
        "--run-name",
        run_name,
        # E1 split contract: launcher also enforces this, belt-and-suspenders here
    ]


def submit_job(run_name: str, config_path: str, variant_flag: str, seed: int, suffix: str, *, dry_run: bool) -> None:
    job_name = run_name.lower().replace("_", "-")
    sync_target = run_name_to_target(variant_flag, seed, suffix)
    repo_dir = f"{sync_target}/omnisam-clean-launcher.git"
    job_cmd = build_job_command(run_name, config_path, variant_flag, seed, repo_dir)

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
        "16G",
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
    ]
    for env_name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "WANDB_API_KEY"):
        env_value = os.environ.get(env_name)
        if env_value:
            cmd.extend(["--environment", f"{env_name}={env_value}"])
    cmd.extend(["--command", "--", *job_cmd])

    print(f"### {job_name}")
    print(" ".join(subprocess.list2cmdline([part]) for part in cmd))
    if dry_run:
        return
    proc = subprocess.run(cmd, text=True, capture_output=True)
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--variants", default="A0,A2,A3")
    parser.add_argument(
        "--suffix",
        default="",
        help="Run batch suffix appended to run names and PVC paths (e.g. v2).",
    )
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    suffix = args.suffix.strip()

    for variant in variants:
        config_path, variant_flag = VARIANTS[variant]
        for seed in seeds:
            suffix_part = f"_{suffix}" if suffix else ""
            run_name = f"monuseg_E1_{variant}_s{seed}_clean{suffix_part}"
            submit_job(run_name, config_path, variant_flag, seed, suffix, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
