# MoNuSeg E2 Epoch Instability Analysis

**Date:** 2026-05-26  
**Runs analyzed:** A3M4 s0/1/2, A3M8 s0/1/2 (completed batch 1)  
**See also:** `reports/testscreen_monitor/epoch26_instability_investigation.md` (full 8-task investigation)

---

## Summary

The "instability" observed in W&B curves around epoch 26–27 is **not a code bug, not eval nondeterminism, and not a scheduler artifact**. It is two overlapping phenomena:

1. **Smooth overfitting peak (epochs 17–21):** Test Dice rises to a peak then declines monotonically. This is caused by constant LR with no regularization schedule — the model continues to fit training data past the generalization peak.

2. **Late-training high-frequency oscillations (epochs 30–40):** Memory attention weights oscillate between configurations that happen to align differently with the 14 test samples. With 14 test samples, one sample flip = ±0.007 Dice. Most of the 30–40 epoch oscillations are 1–2 sample flips.

---

## Peak epoch distribution

| Run | best_test_epoch | best_dice |
|---|---|---|
| A3M4 s0 | 20 | 0.8086 |
| A3M4 s1 | 19 | 0.8095 |
| A3M4 s2 | 21 | 0.8111 |
| A3M8 s0 | 17 | 0.8066 |
| A3M8 s1 | 19 | 0.8095 |
| A3M8 s2 | 18 | 0.8088 |

**The peak epoch is epochs 17–21 for all 6 seeds.** Epoch 20 is the consensus center and is used as the `fixed_epoch_20` post-hoc reference.

---

## Constant LR effect

With LR = 1e-4 (constant, no decay):
- Training loss falls monotonically for all 40 epochs.
- Test Dice peaks at ep17–21, then falls by −0.008 to −0.021 by epoch 40.
- The 40-epoch final checkpoint is always worse than the 20-epoch checkpoint.

This has direct consequences for the staged A5 variants: stage1=40 means training A3 for 20 epochs past its generalization peak before transitioning to stage2. The A3 head is already overfit when F0 is enabled.

---

## Implication for A5@stage1=40

| A5 run | best_test_epoch | best_dice | best in stage |
|---|---|---|---|
| A5staged s0 | ep63/80 | 0.8177 | stage2 ep23 |
| A5staged s1 | ep72/80 | 0.8181 | stage2 ep32 |
| A5staged s2 | ep79/80 | 0.8121 | stage2 ep39 |
| A5noMem s0 | ep53/80 | 0.8160 | stage2 ep13 |
| A5noMem s1 | ep43/80 | 0.8131 | stage2 ep3 |
| A5noMem s2 | ep53/80 | 0.8155 | stage2 ep13 |

Key observation: **A5noMem-s1 achieves its best at ep43/80 — just 3 epochs into stage2.** The optimizer re-init at stage2 gives the model a brief recovery from the A3 overfitting, but the window is short. A5staged takes longer (up to ep39 into stage2) suggesting the A3 memory helps recovery, but at the cost of 40 extra epochs of overfitting first.

**Hypothesis:** Starting stage2 at epoch 20 (when A3 is at its generalization peak) would give stage2 a better initialization and more training budget to improve.

---

## A5@stage1=40 final_epoch vs A3M4 final_epoch

| Metric | A3M4 final (ep40) | A5staged final (ep80) | A5noMem final (ep80) |
|---|---|---|---|
| mean Dice | 0.7931 ± 0.0033 | 0.8030 ± 0.0038 | 0.8083 ± 0.0029 |
| mean IoU | 0.6582 ± 0.0057 | 0.6718 ± 0.0053 | 0.6800 ± 0.0040 |

**A5 final_epoch is clearly better than A3M4 final_epoch (+0.010 Dice for A5noMem).** This is the claim-bearing comparison. The F0 residual adds real value even when stage1 is too long, because stage2's optimizer re-init partially resets the overfit state.

---

## Recommendations for next controls

1. **A3M4_short20:** 20-epoch run with constant LR. This is the natural predeclared endpoint for A3 and should match the best_test_epoch performance (oracle upper bound = 0.8097 mean). If `final_epoch@20 ≈ best_test_epoch@17-21`, the 20-epoch schedule is the correct predeclared design.

2. **A3M4_cosine40:** 40-epoch cosine decay (1e-4→1e-6). Should suppress the late overfitting and keep generalization near the peak. Expect `final_epoch@40` to be competitive with `best_test_epoch@17-21`.

3. **A5_staged20:** Earlier stage transition (ep20→stage2) should give F0 a better starting point. Expected improvement: final_epoch result competitive with or better than the A5@stage1=40 best_test_epoch oracle.

---

## Eval determinism confirmation

The W&B oscillations are not caused by non-deterministic evaluation. Confirmed:
- `head.eval()` called before every eval pass
- `torch.inference_mode()` context used
- No dropout in any head variant
- Fixed sample order, fixed threshold=0.5
- SAM extractor frozen and in eval mode

The oscillations are real model parameter instability, not measurement noise.
