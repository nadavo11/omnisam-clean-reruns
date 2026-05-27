#!/usr/bin/env python3
"""Submit MoNuSeg 80-epoch layer-warmup ablation jobs via RunAI.

6 variants × 3 seeds = 18 full jobs.
Smoke test: 1 job (variant 1, seed 0, 2 epochs) submitted first.

Run with --dry-run to preview commands without submitting.
Run with --smoke-only to submit only the smoke test.
Run with --skip-smoke to skip smoke test and go straight to full jobs.

Job naming: monuseg-ts80-{short_key}-s{seed}
Group: monuseg_testscreen80_layerwarm_aug
Output PVC: /storage/nada/test_screen_80/{variant_key}/s{seed}
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
WANDB_GROUP = "monuseg_testscreen80_layerwarm_aug"

# --------------------------------------------------------------------------
# Variant registry
# --------------------------------------------------------------------------
# key → (human_name, config_path, short_job_name)
VARIANTS: dict[str, tuple[str, str, str]] = {
    "v1_progressive": (
        "progressive_A2_to_A3M4_to_F0_aug80",
        "configs/test_screen_80_layerwarm/v1_progressive_a2_a3m4_f0_aug80.yaml",
        "monuseg-ts80-v1-prog",
    ),
    "v2_legacy_staged40": (
        "legacy_A5staged40_A3M4_F0_aug80",
        "configs/test_screen_80_layerwarm/v2_legacy_a5staged40_a3m4_f0_aug80.yaml",
        "monuseg-ts80-v2-staged40",
    ),
    "v3_all_refine_mem": (
        "all_refinement_layers_memory_aug80",
        "configs/test_screen_80_layerwarm/v3_all_refinement_layers_memory_aug80.yaml",
        "monuseg-ts80-v3-allrefmem",
    ),
    "v4_no_memory": (
        "A2_to_F0_no_memory_aug80",
        "configs/test_screen_80_layerwarm/v4_a2_to_f0_no_memory_aug80.yaml",
        "monuseg-ts80-v4-nomem",
    ),
    "v5_a3m4_no_f0": (
        "A3M4_no_F0_aug80",
        "configs/test_screen_80_layerwarm/v5_a3m4_no_f0_aug80.yaml",
        "monuseg-ts80-v5-nof0",
    ),
    "v6_onestage_gated": (
        "full_A3M4_F0_from_epoch0_gated_aug80",
        "configs/test_screen_80_layerwarm/v6_full_a3m4_f0_from_epoch0_gated_aug80.yaml",
        "monuseg-ts80-v6-onestage",
    ),
}

SMOKE_VARIANT = "v1_progressive"
SMOKE_SEED = 0
SMOKE_MAX_EPOCHS = 2
SMOKE_LIMIT_TRAIN = 5
SMOKE_LIMIT_TEST = 3


def pvc_output(variant_key: str, seed: int) -> str:
    return f"/storage/nada/test_screen_80/{variant_key}/s{seed}"


# Short alphanumeric codes for sync target paths (must keep container names ≤ 63 chars)
# Container name pattern: git-sync-git--storage-nada-{sync_target_suffix}
# Budget: 63 - len("git-sync-git--storage-nada-") = 63 - 27 = 36 chars for suffix
_VARIANT_SHORT: dict[str, str] = {
    "v1_progressive":    "ts80l/v1p",
    "v2_legacy_staged40": "ts80l/v2s",
    "v3_all_refine_mem": "ts80l/v3r",
    "v4_no_memory":      "ts80l/v4n",
    "v5_a3m4_no_f0":     "ts80l/v5a",
    "v6_onestage_gated": "ts80l/v6g",
}


def submit_job(
    variant_key: str,
    seed: int,
    *,
    smoke: bool = False,
    dry_run: bool,
) -> None:
    human_name, config_path, base_job_name = VARIANTS[variant_key]
    job_name = f"{base_job_name}-s{seed}" + ("-smoke" if smoke else "")
    run_name = job_name.replace("-", "_")
    output_dir = pvc_output(variant_key + ("/smoke" if smoke else ""), seed)
    short = _VARIANT_SHORT[variant_key]
    sync_target = f"/storage/nada/{short}/s{seed}"
    repo_dir = f"{sync_target}/omnisam-clean-launcher.git"

    inner_cmd = [
        "python",
        f"{repo_dir}/launch_monuseg_80_layerwarm.py",
        "--run-name", run_name,
        "--human-name", human_name,
        "--config", config_path,
        "--seed", str(seed),
        "--output-dir", output_dir,
        "--wandb-group", WANDB_GROUP,
        "--wandb-job-type", "TEST_SCREEN_80",
    ]
    if smoke:
        # Smoke test: run inside the omnisam-clean-reruns zip extraction directly
        # using --max-epochs and --limit-* flags on the training script
        # Override: call train_monuseg_80_layerwarm.py directly with overrides
        inner_cmd = [
            "python",
            f"{repo_dir}/launch_monuseg_80_layerwarm.py",
            "--run-name", run_name,
            "--human-name", human_name + "_SMOKE",
            "--config", config_path,
            "--seed", str(seed),
            "--output-dir", output_dir,
            "--wandb-group", WANDB_GROUP,
            "--wandb-job-type", "TEST_SCREEN_80_SMOKE",
        ]
        # Append smoke overrides via env (launcher passes these through)
        # We'll pass them via a wrapper approach — see note below

    outer_cmd = [
        "runai", "submit",
        "--name", job_name,
        "--image", IMAGE,
        "--gpu", "1",
        "--project", PROJECT,
        "--cpu", "4",
        "--memory", "32G",
        "--image-pull-policy", "IfNotPresent",
        "--existing-pvc", "claimname=storage,path=/storage",
        "--working-dir", "/storage/nada",
        "--git-sync",
        f"source={REPO_GIT},branch=main,target={sync_target}",
        "--pod-running-timeout", "20m",
        "--environment", "RUNAI_STORAGE_ROOT=/storage/nada",
        "--environment", "RUNAI_CACHE_ROOT=/storage/nada/cache/texture-representations",
        "--environment", "RUNAI_VENV_ROOT=/storage/nada/envs/texture-representations-runai-cu128",
        "--environment", "RUNAI_ENABLE_PERSISTENT_VENV=1",
        "--environment", "WANDB_MODE=online",
        "--environment", "WANDB_PROJECT=frozen-sam-readout",
    ]
    if smoke:
        outer_cmd.extend([
            "--environment", f"SMOKE_MAX_EPOCHS={SMOKE_MAX_EPOCHS}",
            "--environment", f"SMOKE_LIMIT_TRAIN={SMOKE_LIMIT_TRAIN}",
            "--environment", f"SMOKE_LIMIT_TEST={SMOKE_LIMIT_TEST}",
        ])
    for env_name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "WANDB_API_KEY"):
        val = os.environ.get(env_name)
        if val:
            outer_cmd.extend(["--environment", f"{env_name}={val}"])
    outer_cmd.extend(["--command", "--", *inner_cmd])

    print(f"\n### {job_name} ({'SMOKE' if smoke else 'full'})")
    print(" ".join(subprocess.list2cmdline([p]) for p in outer_cmd))
    if dry_run:
        return
    proc = subprocess.run(outer_cmd, text=True, capture_output=True)
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-only", action="store_true",
                        help="Submit smoke test only (v1 seed0, 2 epochs)")
    parser.add_argument("--skip-smoke", action="store_true",
                        help="Skip smoke test, submit full jobs directly")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument(
        "--variants",
        default=",".join(VARIANTS.keys()),
        help="Comma-separated variant keys (default: all 6)",
    )
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    variant_keys = [v.strip() for v in args.variants.split(",") if v.strip()]
    for key in variant_keys:
        if key not in VARIANTS:
            raise SystemExit(f"Unknown variant '{key}'. Valid: {list(VARIANTS)}")

    if not args.skip_smoke:
        print("\n" + "=" * 60)
        print("STEP 1: Smoke test (v1 seed0, 2 epochs)")
        print("=" * 60)
        submit_job(SMOKE_VARIANT, SMOKE_SEED, smoke=True, dry_run=args.dry_run)

    if args.smoke_only:
        print("\nSmoke-only mode — not submitting full jobs.")
        print("After smoke passes, re-run with --skip-smoke to submit all.")
        return 0

    print("\n" + "=" * 60)
    print(f"STEP 2: Full jobs ({len(variant_keys)} variants × {len(seeds)} seeds)")
    print("=" * 60)
    for key in variant_keys:
        for seed in seeds:
            submit_job(key, seed, smoke=False, dry_run=args.dry_run)

    print(f"\nSubmitted {len(variant_keys) * len(seeds)} jobs.")
    print(f"W&B group: {WANDB_GROUP}")
    print("Monitor: runai list | grep monuseg-ts80")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
