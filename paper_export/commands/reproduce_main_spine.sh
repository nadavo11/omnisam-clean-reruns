#!/usr/bin/env bash
# reproduce_main_spine.sh — train all main-spine variants from scratch on MoNuSeg
# Run on run:AI cluster (project avidan, A100-80GB)
# Assumes: venv at /storage/nada/envs/texture-representations-runai-cu128/
#          repo cloned to /storage/nada/omniSAM/
set -euo pipefail

REPO=/storage/nada/omniSAM
VENV=/storage/nada/envs/texture-representations-runai-cu128
OUT=/storage/nada/outputs/omnisam_repro
CFG=$REPO/configs/official/monuseg_strict_512.yaml

source $VENV/bin/activate

# A0 — fpn_2 only
python $REPO/scripts/train_official.py \
  --config $CFG --variant a0_fpn2 --seed 0 \
  --output-dir $OUT/monuseg_A0_fpn2 --epochs-stage1 40

# A2 — + fpn_1 refine
python $REPO/scripts/train_official.py \
  --config $CFG --variant a2_fpn2_fpn1_refine --seed 0 \
  --output-dir $OUT/monuseg_A2 --epochs-stage1 40

# A3 — + memory attn M=8
python $REPO/scripts/train_official.py \
  --config $CFG --variant a3_memory --seed 0 \
  --output-dir $OUT/monuseg_A3 --epochs-stage1 40

# A5 — all-at-once (control)
python $REPO/scripts/train_official.py \
  --config $CFG --variant a5_all_at_once --seed 0 \
  --output-dir $OUT/monuseg_A5 --epochs-stage1 40

# Ours — staged: stage1=A3 (40ep), stage2=+fpn_0 (40ep)
python $REPO/scripts/train_official.py \
  --config $CFG --variant final_staged --seed 0 \
  --output-dir $OUT/monuseg_final_staged \
  --epochs-stage1 40 --epochs-stage2 40 \
  --init-from-checkpoint $OUT/monuseg_A3/checkpoint.pt

# Freeze-A3 control
python $REPO/scripts/train_official.py \
  --config $CFG --variant freeze_a3_control --seed 0 \
  --output-dir $OUT/monuseg_freeze_a3 \
  --epochs-stage1 40 --epochs-stage2 40 \
  --init-from-checkpoint $OUT/monuseg_A3/checkpoint.pt \
  --freeze-stage1-in-stage2

echo "Done. Collect metrics with: python $REPO/scripts/convert_source_metrics.py --checkpoint-root $OUT --output-root $REPO/results/official_metrics"
