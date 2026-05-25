#!/usr/bin/env bash
# eval_all_official.sh — evaluate all variants and write official_metrics.json
set -euo pipefail

REPO=/storage/nada/omniSAM
VENV=/storage/nada/envs/texture-representations-runai-cu128
CKPT_ROOT=/storage/nada/outputs/fixed_resize_enhancement
OUT=$REPO/results/official_metrics

source $VENV/bin/activate

# Convert source summary.json → official_metrics.json
python $REPO/scripts/convert_source_metrics.py \
  --checkpoint-root $CKPT_ROOT \
  --output-root $OUT

# Generate all paper tables
python $REPO/scripts/make_paper_tables.py \
  --metrics-root $OUT \
  --lock-file $REPO/results/official_result_lock.yaml \
  --output-dir $REPO/paper_assets/tables

echo "Tables written to $REPO/paper_assets/tables"
