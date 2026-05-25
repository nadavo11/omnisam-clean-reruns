# omniSAM: Staged Multiscale Frozen-SAM3 Readout

**Paper story.** Frozen SAM3 contains useful out-of-distribution medical
segmentation evidence. We bypass SAM's prompt / mask-decoder entirely and train
a small staged multiscale readout over the frozen SAM feature pyramid
(`fpn_2`, `fpn_1`, `fpn_0`). The official method, *Staged Multiscale Frozen-SAM3
Readout*, sets a new ablation-controlled bar on MoNuSeg-strict-512 and
GlaS-fixed-224.

## Headline results

| Dataset | Variant | Dice | IoU |
|---|---|---|---|
| MoNuSeg-strict-512 | A0 (`fpn_2` only) | 0.8103 | 0.6835 |
| MoNuSeg-strict-512 | A2 (`fpn_2`+`fpn_1` refine) | 0.8423 | 0.7284 |
| MoNuSeg-strict-512 | A3 (A2 + memory attn) | 0.8339 | 0.7161 |
| MoNuSeg-strict-512 | A5 all-at-once | 0.8124 | 0.6865 |
| MoNuSeg-strict-512 | **Final staged (mean 3 seeds)** | **0.8545 ± 0.0050** | **0.7467 ± 0.0073** |
| MoNuSeg-strict-512 | Freeze-A3 control | 0.8234 | 0.7010 |
| GlaS-fixed-224 | A0 | 0.9374 | 0.8847 |
| GlaS-fixed-224 | A2 (best) | **0.9405** | **0.8897** |
| GlaS-fixed-224 | A3 | 0.9243 | 0.8635 |
| GlaS-fixed-224 | A5-safe | 0.9259 | 0.8656 |

See `RESULTS.md` and `results/official_result_lock.yaml` for the locked
official numbers.

## Quickstart

```bash
pip install -e .
pytest -q
python scripts/audit_protocol.py --config configs/official/monuseg_strict_512.yaml
python scripts/eval_official.py \
    --config configs/official/monuseg_strict_512.yaml \
    --method-config configs/official/method_staged_multiscale.yaml \
    --checkpoint checkpoints/final_staged_seed0/stage2.pt \
    --output results/official_metrics/monuseg/final_staged_seed0
```

See `REPRODUCIBILITY.md` for the full pipeline (training stage-1, stage-2,
official eval, paper tables, comparison grids).

## Repository layout

- `src/frozen_sam_readout/` — package
- `configs/official/` — locked protocols + method config + visuals config
- `configs/research/` — exploratory ablations (NOT paper-headline-safe)
- `results/official_result_lock.yaml` — pinned headline numbers
- `scripts/` — train / eval / visualize / audit
- `tests/` — pytest suite
- `paper_assets/` — generated tables and figures

## Method TL;DR

- Backbone: **frozen** SAM3 image encoder (no prompt encoder, no mask decoder).
- Stage 1: A3 head = (fpn_2 → coarse) + (fpn_1 → residual refine) +
  M = 8 memory-attention tokens, producing `L_A3` (40 epochs MoNuSeg / 200 GlaS).
- Stage 2: F0-learned-alpha residual head adds
  `R0 = F0Residual(fpn_0, up(L_A3))` and predicts
  `L_final = up(L_A3) + alpha * R0`, with `alpha = sigmoid(raw_alpha)`,
  `raw_alpha = log(0.05 / 0.95)` so `alpha ≈ 0.05` at init, and the final 1×1
  conv of `F0Residual` zero-initialized. Trained jointly for +40 epochs; A3 is
  NOT frozen.
- Prediction: `sigmoid(L_final) > 0.5`.
- Metric: `compute_binary_metrics` (verbatim port from the original repo) →
  `direct_foreground_dice` and `direct_foreground_iou`.
