#!/usr/bin/env bash
# make_all_paper_assets.sh — generate all paper tables + visuals in one shot
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)

# Tables
python $REPO/scripts/make_paper_tables.py \
  --metrics-root $REPO/results/official_metrics \
  --lock-file $REPO/results/official_result_lock.yaml \
  --output-dir $REPO/paper_export/tables

# Official visuals — MoNuSeg
python $REPO/scripts/make_official_visuals.py \
  --config $REPO/configs/official/monuseg_strict_512.yaml \
  --checkpoint-dir /storage/nada/outputs/fixed_resize_enhancement/migrated_heads \
  --output-dir $REPO/outputs/official_visuals/monuseg \
  --limit 14 --device cuda

# Official visuals — GlaS
python $REPO/scripts/make_official_visuals.py \
  --config $REPO/configs/official/glas_fixed_224.yaml \
  --checkpoint-dir /storage/nada/outputs/fixed_resize_enhancement/migrated_heads \
  --output-dir $REPO/outputs/official_visuals/glas \
  --limit 10 --device cuda

echo "All paper assets generated."
