#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys


IMAGE = "pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime"
PROJECT = "avidan"
REPO_GIT = "https://github.com/nadavo11/omnisam-clean-reruns.git"

VARIANTS = {
    "FPNplusD0_boxprompt_aug80_reference": "configs/test_screen_learned_prompt/fpnplusd0_boxprompt_reference_aug80.yaml",
    "FPNplusD0_learned_sparse_prompt_aug80": "configs/test_screen_learned_prompt/fpnplusd0_learned_sparse_prompt_aug80.yaml",
    "FPNplusD0_box_plus_learned_delta_aug80": "configs/test_screen_learned_prompt/fpnplusd0_box_plus_learned_delta_aug80.yaml",
    "D0small_box_plus_learned_delta_aug80": "configs/test_screen_learned_prompt/d0small_box_plus_learned_delta_aug80.yaml",
}

JOB_SLUGS = {
    "FPNplusD0_boxprompt_aug80_reference": "monuseg-lprompt-fpnd0-boxref",
    "FPNplusD0_learned_sparse_prompt_aug80": "monuseg-lprompt-fpnd0-sparse",
    "FPNplusD0_box_plus_learned_delta_aug80": "monuseg-lprompt-fpnd0-delta",
    "D0small_box_plus_learned_delta_aug80": "monuseg-lprompt-d0small-delta",
}


def submit(*, variant: str, seed: int, branch: str, dry_run: bool) -> None:
    job_name = f"{JOB_SLUGS[variant]}-s{seed}"
    run_name = job_name.replace("-", "_")
    sync_target = f"/storage/nada/learned_prompt_code/{JOB_SLUGS[variant]}/s{seed}"
    repo_dir = f"{sync_target}/omnisam-clean-reruns.git"
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
        f"source={REPO_GIT},branch={branch},target={sync_target}",
        "--pod-running-timeout",
        "20m",
        "--environment",
        "RUNAI_STORAGE_ROOT=/storage/nada",
        "--environment",
        "RUNAI_CACHE_ROOT=/storage/nada/cache/texture-representations",
        "--environment",
        "RUNAI_VENV_ROOT=/storage/nada/envs/texture-representations-runai-cu128",
        "--environment",
        "WANDB_MODE=online",
        "--environment",
        "WANDB_PROJECT=frozen-sam-readout",
        "--command",
        "--",
        "python",
        f"{repo_dir}/scripts/train_monuseg_test_screen.py",
        "--config",
        f"{repo_dir}/{VARIANTS[variant]}",
        "--seed",
        str(seed),
        "--run-name",
        run_name,
        "--output-dir",
        f"/storage/nada/learned_prompt/{variant}/s{seed}",
        "--wandb-group",
        variant,
        "--wandb-job-type",
        "TEST_SCREEN_LEARNED_PROMPT",
    ]
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
    parser.add_argument("--branch", default="codex/clean-e1-runai-wandb")
    parser.add_argument("--variants", default="FPNplusD0_box_plus_learned_delta_aug80,FPNplusD0_learned_sparse_prompt_aug80,D0small_box_plus_learned_delta_aug80")
    parser.add_argument("--seeds", default="0,1,2")
    args = parser.parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    for variant in variants:
        if variant not in VARIANTS:
            raise SystemExit(f"Unknown variant {variant!r}")
        for seed in seeds:
            submit(variant=variant, seed=seed, branch=args.branch, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
