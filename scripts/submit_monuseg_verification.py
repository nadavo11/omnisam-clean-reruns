#!/usr/bin/env python3
"""Submit MoNuSeg verification + mechanism ablation jobs.

Blocks
------
v6r   — v6 clean resubmission (from-epoch-0 gated, seeds 0/1/2)
a     — Block A: v3 fresh seeds 3/4/5 (stability verification)
b     — Block B: canonical paper-safe v3 rerun (seeds 0/1/2)
c1    — C1 abrupt F0 (seeds 0/1/2 by default, or use --c-seeds to restrict)
c2    — C2 gated no zero-init
c3    — C3 F0 memory off
c4    — C4 F0 memory only (coarse/F1 memory disabled)

Usage examples
--------------
# Dry-run everything
python scripts/submit_monuseg_verification.py --dry-run

# Submit v6 resub only (smoke first, then full)
python scripts/submit_monuseg_verification.py --block v6r

# Submit block A + B
python scripts/submit_monuseg_verification.py --block a b

# Submit all blocks (default: block C uses 1 seed per ablation)
python scripts/submit_monuseg_verification.py --block v6r a b c1 c2 c3 c4

# Submit block C with 3 seeds
python scripts/submit_monuseg_verification.py --block c1 c2 c3 c4 --c-seeds 0,1,2

# Skip smoke test
python scripts/submit_monuseg_verification.py --block v6r --skip-smoke
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

IMAGE = "pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime"
PROJECT = "avidan"
REPO_GIT = "https://github.com/nadavo11/omnisam-clean-launcher.git"

# W&B groups per block
_WANDB_GROUP = {
    "v6r": "monuseg_testscreen80_layerwarm_aug",
    "a":   "monuseg_testscreen80_layerwarm_aug_verification",
    "b":   "monuseg_testscreen80_layerwarm_aug_verification",
    "c1":  "monuseg_testscreen80_layerwarm_aug_verification",
    "c2":  "monuseg_testscreen80_layerwarm_aug_verification",
    "c3":  "monuseg_testscreen80_layerwarm_aug_verification",
    "c4":  "monuseg_testscreen80_layerwarm_aug_verification",
}

# (human_name, config_rel_path, base_job_prefix, short_sync_key)
# short_sync_key → sync_target = /storage/nada/{short_sync_key}/s{seed}
# Container name: git-sync-git--storage-nada-{short_sync_key_dashes}-s{seed}
# Budget: 63 - 27 = 36 chars.  All below verified ≤ 36 chars after dash-replace.
BLOCKS: dict[str, tuple[str, str, str, str]] = {
    "v6r": (
        "full_A3M4_F0_from_epoch0_gated_aug80",
        "configs/test_screen_80_layerwarm_verify/v6_resub_aug80.yaml",
        "monuseg-ts80-v6r",
        "ts80v/v6r",
    ),
    "a": (
        "all_refinement_layers_memory_aug80_fresh345",
        "configs/test_screen_80_layerwarm_verify/block_a_v3_fresh_345.yaml",
        "monuseg-vera-v3",
        "vera/v3a",
    ),
    "b": (
        "official_candidate_gated_all_refine_mem_F0_aug80_canonical",
        "configs/test_screen_80_layerwarm_verify/block_b_canonical_gated_all_refine_mem.yaml",
        "monuseg-verb-v3",
        "verb/v3b",
    ),
    "c1": (
        "all_refine_mem_abrupt_F0_no_gate",
        "configs/test_screen_80_layerwarm_verify/block_c1_abrupt_f0_aug80.yaml",
        "monuseg-abl-c1",
        "ablc/c1",
    ),
    "c2": (
        "all_refine_mem_gated_but_no_zero_init",
        "configs/test_screen_80_layerwarm_verify/block_c2_gated_no_zero_init_aug80.yaml",
        "monuseg-abl-c2",
        "ablc/c2",
    ),
    "c3": (
        "memory_mid_only_F0_no_memory",
        "configs/test_screen_80_layerwarm_verify/block_c3_f0_no_f0memory_aug80.yaml",
        "monuseg-abl-c3",
        "ablc/c3",
    ),
    "c4": (
        "memory_F0_only_mid_no_memory",
        "configs/test_screen_80_layerwarm_verify/block_c4_f0memory_only_aug80.yaml",
        "monuseg-abl-c4",
        "ablc/c4",
    ),
}

# Default seeds per block
DEFAULT_SEEDS = {
    "v6r": [0, 1, 2],
    "a":   [3, 4, 5],
    "b":   [0, 1, 2],
    # Block C defaults to 1 seed; expand to 3 if informative
    "c1":  [0],
    "c2":  [0],
    "c3":  [0],
    "c4":  [0],
}

# W&B run-name format per block
def _run_name(block: str, seed: int) -> str:
    if block == "v6r":
        return f"monuseg_testscreen80_v6_from_epoch0_gated_resub_s{seed}"
    if block == "a":
        return f"monuseg_verify_v3_all_refine_mem_s{seed}"
    if block == "b":
        return f"monuseg_canon_v3_all_refine_mem_s{seed}"
    return f"monuseg_abl_{block}_s{seed}"


def _output_dir(block: str, seed: int) -> str:
    paths = {
        "v6r": f"/storage/nada/verify/v6_resub/s{seed}",
        "a":   f"/storage/nada/verify/block_a_v3_fresh/s{seed}",
        "b":   f"/storage/nada/verify/block_b_canonical/s{seed}",
        "c1":  f"/storage/nada/verify/block_c1_abrupt_f0/s{seed}",
        "c2":  f"/storage/nada/verify/block_c2_no_zero_init/s{seed}",
        "c3":  f"/storage/nada/verify/block_c3_f0_no_mem/s{seed}",
        "c4":  f"/storage/nada/verify/block_c4_f0mem_only/s{seed}",
    }
    return paths[block]


SMOKE_MAX_EPOCHS = 2
SMOKE_LIMIT_TRAIN = 5
SMOKE_LIMIT_TEST = 3


def _submit_job(
    block: str,
    seed: int,
    *,
    smoke: bool = False,
    dry_run: bool,
) -> None:
    human_name, config_path, base_prefix, short = BLOCKS[block]
    wandb_group = _WANDB_GROUP[block]
    run_name = _run_name(block, seed)
    job_name = f"{base_prefix}-s{seed}" + ("-smoke" if smoke else "")
    output_dir = _output_dir(block, seed) + ("/smoke" if smoke else "")
    sync_target = f"/storage/nada/{short}/s{seed}"
    repo_dir = f"{sync_target}/omnisam-clean-launcher.git"
    wandb_job_type = "VERIFY_SMOKE" if smoke else "VERIFY"

    inner_cmd = [
        "python", f"{repo_dir}/launch_monuseg_80_layerwarm.py",
        "--run-name",    run_name + ("_SMOKE" if smoke else ""),
        "--human-name",  human_name + ("_SMOKE" if smoke else ""),
        "--config",      config_path,
        "--seed",        str(seed),
        "--output-dir",  output_dir,
        "--wandb-group", wandb_group,
        "--wandb-job-type", wandb_job_type,
    ]

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
        "--git-sync", f"source={REPO_GIT},branch=main,target={sync_target}",
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

    print(f"\n### {job_name} ({'SMOKE' if smoke else 'full'}) — {human_name}")
    print(" ".join(subprocess.list2cmdline([p]) for p in outer_cmd))
    if dry_run:
        return
    proc = subprocess.run(outer_cmd, text=True, capture_output=True)
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true",
                        help="Skip smoke test (v6r only does smoke by default)")
    parser.add_argument(
        "--block", nargs="+", default=list(BLOCKS.keys()),
        choices=list(BLOCKS.keys()),
        help="Which blocks to submit (default: all)",
    )
    parser.add_argument(
        "--c-seeds", default=None,
        help="Override seeds for block C ablations (e.g. '0,1,2'). "
             "Default: 1 seed (seed 0) per C ablation.",
    )
    parser.add_argument(
        "--seeds", default=None,
        help="Override seeds for ALL blocks (comma-separated). "
             "Overrides --c-seeds and per-block defaults.",
    )
    args = parser.parse_args()

    blocks = args.block
    c_override = [int(s) for s in args.c_seeds.split(",") if s.strip()] if args.c_seeds else None
    global_override = [int(s) for s in args.seeds.split(",") if s.strip()] if args.seeds else None

    # Smoke test: only for v6r
    if "v6r" in blocks and not args.skip_smoke:
        print("\n" + "=" * 60)
        print("SMOKE: v6r seed 0 (2 epochs)")
        print("=" * 60)
        _submit_job("v6r", seed=0, smoke=True, dry_run=args.dry_run)
        print("\nSmoke submitted. Verify it passes before full jobs run.")
        print("Full jobs will still be submitted now — RunAI queues them.")
        print("If smoke fails, cancel remaining jobs with:")
        print("  runai list -p avidan | grep monuseg | awk '{print $1}' | "
              "xargs -I{} runai delete job {} -p avidan")

    # Full jobs
    for block in blocks:
        if global_override:
            seeds = global_override
        elif block in ("c1", "c2", "c3", "c4") and c_override:
            seeds = c_override
        else:
            seeds = DEFAULT_SEEDS[block]

        print("\n" + "=" * 60)
        print(f"Block {block.upper()} — {BLOCKS[block][0]} — seeds {seeds}")
        print("=" * 60)
        for seed in seeds:
            _submit_job(block, seed, smoke=False, dry_run=args.dry_run)

    total = sum(
        len(global_override or (c_override if b in ("c1","c2","c3","c4") else None) or DEFAULT_SEEDS[b])
        for b in blocks
    )
    print(f"\nSubmitted {total} full jobs across blocks: {blocks}")
    print("Monitor: runai list -p avidan | grep -E 'monuseg-ts80-v6r|monuseg-ver|monuseg-abl'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
