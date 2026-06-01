#!/usr/bin/env python3
"""Submit the MoNuSeg task/spatial residual batch through RunAI."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


IMAGE = "pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime"
PROJECT = "avidan"
WANDB_PROJECT = "frozen-sam-readout"
WANDB_GROUP = "monuseg_d0fpn_task_spatial_residual_aug80"
OUTPUT_ROOT = "/storage/nada/outputs/monuseg_d0fpn_task_spatial_residual_batch"
CODE_ROOT = "/storage/nada/taskspatial_code"
REPO_GIT = "https://github.com/nadavo11/omnisam-clean-reruns.git"
REPO_BRANCH = "codex/clean-e1-runai-wandb"
REPO_DIR_NAME = "omnisam-clean-reruns.git"


VARIANTS: dict[str, tuple[str, str, str]] = {
    "task_tokens_film_spatial_after": (
        "configs/test_screen_task_spatial_residual/monuseg_task_tokens_film_spatial_after_aug80.yaml",
        "Task tokens FiLM then spatial attention",
        "d0fpn_task_tokens_film_spatial_after",
    ),
    "spatial_before_task_tokens_film": (
        "configs/test_screen_task_spatial_residual/monuseg_spatial_before_task_tokens_film_aug80.yaml",
        "Spatial attention then task tokens FiLM",
        "d0fpn_spatial_before_task_tokens_film",
    ),
    "task_conditioned_spatial_attention": (
        "configs/test_screen_task_spatial_residual/monuseg_task_conditioned_spatial_attention_aug80.yaml",
        "Task-conditioned spatial attention",
        "d0fpn_task_conditioned_spatial_attention",
    ),
    "residual_task_spatial_delta": (
        "configs/test_screen_task_spatial_residual/monuseg_residual_task_spatial_delta_aug80.yaml",
        "Residual task-spatial refinement branch",
        "d0fpn_residual_task_spatial_delta",
    ),
    "logit_residual_task_spatial": (
        "configs/test_screen_task_spatial_residual/monuseg_logit_residual_task_spatial_aug80.yaml",
        "Logit-level task-spatial correction",
        "d0fpn_logit_residual_task_spatial",
    ),
    "dual_skip_task_spatial_fusion": (
        "configs/test_screen_task_spatial_residual/monuseg_dual_skip_task_spatial_fusion_aug80.yaml",
        "Dual skip task-spatial fusion",
        "d0fpn_dual_skip_task_spatial_fusion",
    ),
    "source_spatial_gates_task_tokens": (
        "configs/test_screen_task_spatial_residual/monuseg_source_spatial_gates_task_tokens_aug80.yaml",
        "Task-token source-specific spatial gates",
        "d0fpn_source_spatial_gates_task_tokens",
    ),
    "f0_boundary_spatial_task_refine": (
        "configs/test_screen_task_spatial_residual/monuseg_f0_boundary_spatial_task_refine_aug80.yaml",
        "F0 boundary-focused task-spatial refinement",
        "d0fpn_f0_boundary_spatial_task_refine",
    ),
}


def _output_dir(variant: str, seed: int) -> str:
    return f"{OUTPUT_ROOT}/{variant}/s{seed}"


def submit_job(
    variant: str,
    seed: int,
    *,
    dry_run: bool,
    repo_branch: str,
    repo_git: str,
) -> None:
    if variant not in VARIANTS:
        raise SystemExit(f"Unknown variant {variant!r}. Valid: {sorted(VARIANTS)}")
    config_path, human_name, trainer_variant = VARIANTS[variant]
    job_name = f"monuseg-taskspatial-{variant.replace('_', '-')}-s{seed}"
    run_name = f"monuseg_taskspatial_{variant}_s{seed}"
    output_dir = _output_dir(variant, seed)
    repo_target = f"{CODE_ROOT}/{REPO_DIR_NAME}"
    job_cmd = (
        "set -euo pipefail; "
        "if [ -f /storage/nada/envs/texture-representations-runai-cu128/bin/activate ]; then "
        "source /storage/nada/envs/texture-representations-runai-cu128/bin/activate; "
        "fi; "
        "if [ -s /storage/nada/secrets/wandb_api_key ]; then "
        "export WANDB_API_KEY=\"$(cat /storage/nada/secrets/wandb_api_key)\"; "
        "fi; "
        f"cd {repo_target}; "
        "export PYTHONPATH=src; "
        "python scripts/train_monuseg_test_screen.py "
        f"--config {config_path} "
        f"--variant {trainer_variant} "
        f"--seed {seed} "
        f"--run-name {run_name} "
        f"--output-dir {output_dir} "
        "--device cuda "
        f"--wandb-project {WANDB_PROJECT} "
        f"--wandb-group {WANDB_GROUP} "
        "--wandb-job-type TASK_SPATIAL_RESIDUAL_AUG80"
    )

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
        f"source={repo_git},branch={repo_branch},target={CODE_ROOT}",
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
        f"WANDB_PROJECT={WANDB_PROJECT}",
    ]
    for env_name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "WANDB_API_KEY"):
        env_value = os.environ.get(env_name)
        if env_value:
            cmd.extend(["--environment", f"{env_name}={env_value}"])
    cmd.extend(["--command", "--", "bash", "-lc", job_cmd])

    print(f"\n### {job_name}")
    print(f"# {human_name}")
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
    parser.add_argument("--variants", default=",".join(VARIANTS), help="Comma-separated variant keys")
    parser.add_argument("--seeds", default="0,1,2", help="Comma-separated seeds")
    parser.add_argument("--repo-git", default=REPO_GIT, help="Git repo source for RunAI git-sync")
    parser.add_argument("--repo-branch", default=REPO_BRANCH, help="Git branch for RunAI git-sync")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    for variant in variants:
        if variant not in VARIANTS:
            raise SystemExit(f"Unknown variant {variant!r}. Valid: {sorted(VARIANTS)}")

    for variant in variants:
        for seed in seeds:
            submit_job(
                variant,
                seed,
                dry_run=args.dry_run,
                repo_branch=args.repo_branch,
                repo_git=args.repo_git,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
