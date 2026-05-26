# Clean E1 Ablation Results — MoNuSeg

Generated: 2026-05-26  
Branch: codex/clean-e1-runai-wandb  
Commits: 272cde4 (SAM3 HF fallback) · 946590a (test eval) · ec2f540 (channel probe)  
Backbone: facebook/sam3 via transformers.AutoModel fallback (NOT paper-equivalent; official sam3 pkg unavailable on cluster)

---

## Final Results — ALL COMPLETE 🟢

### A0 — fpn_2 only (baseline)

| Seed | Best Val Dice | Best Val IoU | Best Epoch | Test Dice | Test IoU |
|---:|---:|---:|---:|---:|---:|
| s0 | 0.6997 | 0.5435 | 6 | 0.7390 | 0.5869 |
| s1 | 0.7007 | 0.5447 | 5 | 0.7381 | 0.5858 |
| s2 | 0.7019 | 0.5462 | 5 | 0.7401 | 0.5883 |
| **mean** | **0.7008** | **0.5448** | — | **0.7391** | **0.5870** |
| **std** | 0.0009 | 0.0011 | — | 0.0008 | 0.0010 |

### A2 — fpn_2 coarse + fpn_1 residual refine

| Seed | Best Val Dice | Best Val IoU | Best Epoch | Test Dice | Test IoU |
|---:|---:|---:|---:|---:|---:|
| s0 | 0.7810 | 0.6418 | 20 | 0.7999 | 0.6674 |
| s1 | 0.7789 | 0.6390 | 37 | 0.7984 | 0.6654 |
| s2 | 0.7752 | 0.6343 | 35 | 0.7996 | 0.6669 |
| **mean** | **0.7784** | **0.6384** | — | **0.7993** | **0.6666** |
| **std** | 0.0024 | 0.0031 | — | 0.0006 | 0.0008 |

### A3 — A2 + global memory cross-attention

| Seed | Best Val Dice | Best Val IoU | Best Epoch | Test Dice | Test IoU |
|---:|---:|---:|---:|---:|---:|
| s0 | 0.7818 | 0.6428 | 39 | 0.8011 | 0.6691 |
| s1 | 0.7798 | 0.6404 | 34 | 0.8055 | 0.6752 |
| s2 | 0.7862 | 0.6489 | 34 | 0.8043 | 0.6735 |
| **mean** | **0.7826** | **0.6440** | — | **0.8036** | **0.6726** |
| **std** | 0.0026 | 0.0035 | — | 0.0019 | 0.0025 |

---

## Summary Table

| Variant | Val Dice (mean±std) | Val IoU (mean±std) | Test Dice (mean±std) | Test IoU (mean±std) |
|---|---:|---:|---:|---:|
| A0 (fpn_2 only) | 0.7008 ± 0.0009 | 0.5448 ± 0.0011 | 0.7391 ± 0.0008 | 0.5870 ± 0.0010 |
| A2 (+ fpn_1 refine) | 0.7784 ± 0.0024 | 0.6384 ± 0.0031 | 0.7993 ± 0.0006 | 0.6666 ± 0.0008 |
| A3 (+ memory attn) | 0.7826 ± 0.0026 | 0.6440 ± 0.0035 | 0.8036 ± 0.0019 | 0.6726 ± 0.0025 |

---

## Notes

- All results use the transformers.AutoModel SAM-3 fallback (official sam3 pkg absent on cluster).
  The fallback returns 256 channels for ALL fpn levels (vs official: fpn_2=256, fpn_1=64, fpn_0=32).
  A2/A3 heads were therefore built with f1_channels=256 (not 64 as in the paper design).
  Results are consistent across seeds but NOT directly comparable to paper numbers with official SAM-3.
- Checkpoint selection: best val_dice epoch.
- Test set: 14 held-out MoNuSeg crops (fixed split, same for all variants/seeds).
- Val set: fixed 14-sample subset of training split (monuseg_fixed_sample_ids.json).

## Fixes Applied During This Run

1. **ec2f540** — Probe actual backbone channels before building head.
   HF SAM-3 returns fpn_1=256ch; hardcoded 64 caused A2/A3 to crash on first forward pass.
2. **946590a** — Real test eval after best-val checkpoint selection.
   A0 first run wrote a stub; all 9 final runs now write real eval_metrics_test.json.
3. **272cde4** — SAM3 HF fallback vision extraction fix (Sam3VideoModel wraps encoder under detector_model).

## Raw Metrics Files (on cluster PVC)

Pattern: `/storage/nada/ce1/<variant>/s<seed>/.worktrees/<sha>/_work/monuseg_E1_<VARIANT>_s<seed>_clean/repo/outputs/monuseg_E1_<VARIANT>_s<seed>_clean/eval_metrics_{val,test}.json`

Examples:
- `/storage/nada/ce1/a0/s0/.worktrees/f6fe194.../outputs/monuseg_E1_A0_s0_clean/eval_metrics_val.json`
- `/storage/nada/ce1/a2/s0/.worktrees/f6fe194.../outputs/monuseg_E1_A2_s0_clean/eval_metrics_test.json`
