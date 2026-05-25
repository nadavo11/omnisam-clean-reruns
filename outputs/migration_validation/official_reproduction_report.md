# Official Metric Reproduction Report

**Date**: 2026-05-25  
**Repo**: omniSAM (migrated)  
**Evaluator**: `scripts/eval_official.py` → `direct_foreground_dice` / `direct_foreground_iou`  
**Metric source**: `src/frozen_sam_readout/evaluation/metrics.py::compute_binary_metrics`

---

## Status

| Dataset | Variant | Reproduction run | Status |
|---|---|---|---|
| MoNuSeg | A0 | Pending — checkpoint on cluster | ⏳ |
| MoNuSeg | Final staged (seed 0) | Pending — checkpoint on cluster | ⏳ |
| GlaS | A0 | Pending — checkpoint on cluster | ⏳ |
| GlaS | A2 (best) | Pending — checkpoint on cluster | ⏳ |

All official checkpoints reside at `/storage/nada/outputs/fixed_resize_enhancement/` on the
Run:AI cluster. They are not available locally. Reproduction must be run on the cluster.

---

## Locked Expected Metrics

### MoNuSeg (512×512, test split, direct foreground Dice/IoU)

| Variant | Expected Dice | Expected IoU | Source job |
|---|---:|---:|---|
| A0 fpn_2_only | 0.8103 | 0.6835 | ce-monu-a0-2930351 |
| A2 fpn_2+fpn_1_refine | 0.8423 | 0.7284 | ce-monu-a2-2930351 |
| A3 memory M=8 | 0.8339 | 0.7161 | p3-monu-a3m8-ds-noop-416fcc0 |
| A5 all-at-once | 0.8124 | 0.6865 | ce-monu-a5-v2-2930351 |
| Final staged +40ep (seed 0) | 0.8539 | 0.7459 | phase5 seed 0 |
| Final staged +40ep (seed 1) | 0.8499 | 0.7399 | phase5 seed 1 |
| Final staged +40ep (seed 2) | 0.8596 | 0.7544 | phase5 seed 2 |
| **Final staged mean ± std** | **0.8545 ± 0.0050** | **0.7467 ± 0.0073** | 3 seeds |
| Freeze-A3 control | 0.8234 | 0.7010 | phase5 ablation |

### GlaS (224×224, test split, direct foreground Dice/IoU)

| Variant | Expected Dice | Expected IoU | Notes |
|---|---:|---:|---|
| A0 fpn_2_only | 0.9374 | 0.8847 | autosam_dense_v1 aug |
| A2 fpn_2+fpn_1_refine | 0.9405 | 0.8897 | best single variant |
| A3 memory M=8 | 0.9243 | 0.8635 | phase 4 |
| A5-safe learned-α | 0.9259 | 0.8656 | phase 4 |

---

## Reproduction Protocol

```bash
# On cluster — example for MoNuSeg A0
python scripts/eval_official.py \
  --config configs/official/monuseg_strict_512.yaml \
  --method-config configs/official/method_staged_multiscale.yaml \
  --checkpoint /storage/nada/outputs/fixed_resize_enhancement/monuseg_A0_fpn2_d128_512_e40/checkpoint.pt \
  --output outputs/official/monuseg/A0_repro \
  --device cuda

# MoNuSeg Final staged (seed 0)
python scripts/eval_official.py \
  --config configs/official/monuseg_strict_512.yaml \
  --method-config configs/official/method_staged_multiscale.yaml \
  --checkpoint /storage/nada/outputs/fixed_resize_enhancement/monuseg_A5lrn_staged_from_A3s0_e40/checkpoint.pt \
  --output outputs/official/monuseg/final_staged_s0_repro \
  --device cuda

# GlaS A0
python scripts/eval_official.py \
  --config configs/official/glas_fixed_224.yaml \
  --method-config configs/official/method_staged_multiscale.yaml \
  --checkpoint /storage/nada/outputs/fixed_resize_enhancement/glas_A0_fpn2_d128_224_e200/checkpoint.pt \
  --output outputs/official/glas/A0_repro \
  --device cuda
```

---

## Pass/Fail Threshold

Reproduction is considered **PASSED** if:

```
|reproduced_dice - expected_dice| ≤ 0.002
|reproduced_iou  - expected_iou|  ≤ 0.003
```

These tolerances account for:
- Floating-point non-determinism across PyTorch versions
- Dataset loading order variation
- Minor image decoding differences

Any delta exceeding these thresholds requires root-cause investigation before paper freeze.

---

## Reproduction Results (to be filled after cluster run)

### MoNuSeg A0

| Metric | Expected | Reproduced | Delta | Pass? |
|---|---:|---:|---:|---|
| direct_foreground_dice | 0.8103 | — | — | ⏳ |
| direct_foreground_iou | 0.6835 | — | — | ⏳ |

### MoNuSeg Final staged seed 0

| Metric | Expected | Reproduced | Delta | Pass? |
|---|---:|---:|---:|---|
| direct_foreground_dice | 0.8539 | — | — | ⏳ |
| direct_foreground_iou | 0.7459 | — | — | ⏳ |

### GlaS A0

| Metric | Expected | Reproduced | Delta | Pass? |
|---|---:|---:|---:|---|
| direct_foreground_dice | 0.9374 | — | — | ⏳ |
| direct_foreground_iou | 0.8847 | — | — | ⏳ |

---

## Known Preconditions for Reproduction

1. **SAM-3 extractor**: official `sam3` package must be installed (see `sam3_feature_equivalence.md`).
2. **Checkpoint compatibility**: migrated `load_checkpoint` must correctly map state_dict keys from the source-repo format.
3. **Dataset split**: MoNuSeg test split = 14 WSI patches (standard test set from `RationAI/MoNuSeg`); GlaS test split = 80 gland images.
4. **Resize protocol**: BILINEAR PIL resize applied before SAM extraction (`Image.resize((W, H), Image.BILINEAR)`).
5. **No augmentation at eval**: `autosam_dense_v1` is train-only; eval uses the clean resized image.
6. **Threshold**: 0.5 sigmoid threshold.
7. **Metric**: macro-averaged per-sample foreground Dice and IoU via `compute_binary_metrics`.

---

## Checkpoint Key Mapping

The source-repo checkpoints store the head model under keys matching the source
`FrozenSamMultiscaleMaskHead` class. The migrated heads use different class names
but the same parameter structure. If `load_checkpoint` fails with key mismatches,
add a key-remapping table to `src/frozen_sam_readout/training/checkpointing.py`.

**Action required**: run one trial load on the cluster and log any unmapped keys before paper freeze.

---

## Files Written by eval_official.py

For each reproduction run, the script writes:
- `official_metrics.json` — locked schema with headline metrics
- `per_sample_metrics.csv` — per-sample Dice/IoU/TP/FP/FN for error analysis
- `run_manifest.json` — config, checkpoint, timestamp

These are the authoritative outputs for the paper tables.
