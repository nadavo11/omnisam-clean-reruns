#!/usr/bin/env python3
"""Submit the final MoNuSeg D0+FPN capacity sweep through RunAI."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


IMAGE = "pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime"
PROJECT = "avidan"
WANDB_PROJECT = "frozen-sam-readout"
WANDB_GROUP = "monuseg_d0fpn_final_capacity160"
OUTPUT_ROOT = "/storage/nada/outputs/monuseg_d0fpn_final_capacity160_batch"
CODE_ROOT = "/storage/nada/final_capacity160_code"
REPO_DIR_NAME = "omnisam-clean-reruns.git"
REPO_GIT = "https://github.com/nadavo11/omnisam-clean-reruns.git"
REPO_BRANCH = "codex/clean-e1-runai-wandb"


VARIANTS: dict[str, tuple[str, str, str]] = {
    "task_tokens_16_film_wide": (
        "configs/test_screen_final_capacity160/monuseg_cap160_task_tokens_16_film_wide.yaml",
        "Task tokens 16 + wide FiLM",
        "d0fpn_cap160_task_tokens_16_film_wide",
    ),
    "task_tokens_32_film_wide": (
        "configs/test_screen_final_capacity160/monuseg_cap160_task_tokens_32_film_wide.yaml",
        "Task tokens 32 + wide FiLM",
        "d0fpn_cap160_task_tokens_32_film_wide",
    ),
    "task_spatial_residual_2layer": (
        "configs/test_screen_final_capacity160/monuseg_cap160_task_spatial_residual_2layer.yaml",
        "Task-conditioned spatial residual 2-layer",
        "d0fpn_cap160_task_spatial_residual_2layer",
    ),
    "task_spatial_residual_4layer": (
        "configs/test_screen_final_capacity160/monuseg_cap160_task_spatial_residual_4layer.yaml",
        "Task-conditioned spatial residual 4-layer",
        "d0fpn_cap160_task_spatial_residual_4layer",
    ),
    "fine_map_fusion_highres_head": (
        "configs/test_screen_final_capacity160/monuseg_cap160_fine_map_fusion_highres_head.yaml",
        "Fine-map fusion high-res head",
        "d0fpn_cap160_fine_map_fusion_highres_head",
    ),
    "wide_fusion_dim256": (
        "configs/test_screen_final_capacity160/monuseg_cap160_wide_fusion_dim256.yaml",
        "Wide D0+FPN fusion dim256",
        "d0fpn_cap160_wide_fusion_dim256",
    ),
    "deep_conv_fusion_residual": (
        "configs/test_screen_final_capacity160/monuseg_cap160_deep_conv_fusion_residual.yaml",
        "Deep residual conv fusion head",
        "d0fpn_cap160_deep_conv_fusion_residual",
    ),
    "hybrid_best_task_spatial_fine": (
        "configs/test_screen_final_capacity160/monuseg_cap160_hybrid_best_task_spatial_fine.yaml",
        "Hybrid best task-spatial-fine",
        "d0fpn_cap160_hybrid_best_task_spatial_fine",
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
    job_name = f"monuseg-cap160-{variant.replace('_', '-')}-s{seed}"
    run_name = f"monuseg_cap160_{variant}_s{seed}"
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
        "--wandb-job-type FINAL_CAPACITY_160"
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
