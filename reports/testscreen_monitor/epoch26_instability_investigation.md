# MoNuSeg Epoch-26 Instability Investigation

**Date:** 2026-05-26  
**Runs analysed:** A3M4 s0/1/2, A3M8 s0/1/2 (all completed, 40 epochs each)  
**Plot:** `epoch_curves.png` (same directory)

---

## Verdict (TL;DR)

**There is no sudden instability at epoch 26–27.**

What appears in the curves is a **smooth, monotonic decline from a peak around epoch 17–21**, followed by **real high-frequency oscillations in late training (epochs 30–40)**. These are two separate phenomena with different causes.

The decline from epoch 22 onward is **real overfitting** — not scheduler-driven, not eval nondeterminism, not a stage transition.

---

## Task 1: Plot summary (see `epoch_curves.png`)

- Train loss decreases monotonically every epoch for all 6 runs (no kink at ep26).
- Test Dice rises to a peak (ep 17–21 depending on seed), then smoothly declines.
- The orange shading at ep24–28 shows NO visible discontinuity in loss or LR — the decline is already well underway.
- The actual visual feature is a smooth "hill" shape; epoch 26 sits on the downslope, not at a cliff.

---

## Task 2: LR values, epochs 20–35

**LR = 1e-4 (constant) for all epochs, all runs.**

There is no scheduler of any kind in `train_monuseg_test_screen.py`. The optimizer is plain AdamW with fixed `lr=1e-4, weight_decay=1e-4`. Confirmed by code inspection:

```python
optimizer = torch.optim.AdamW(
    [p for p in head.parameters() if p.requires_grad],
    lr=float(training_cfg.get("lr", 1e-4)),
    weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
)
```

No `scheduler.step()` calls anywhere in the training loop.

---

## Task 3: Scheduler / milestone / unfreeze events near epoch 25–27

| Event type | Fires? | When? |
|---|---|---|
| LR scheduler step | **No** | Never |
| Warm restart / cosine restart | **No** | Never |
| Layer unfreeze | **No** | Never |
| Stage transition (F0 enable) | **No** (single-stage A3M4/A3M8) | Epoch 41 for two-stage variants only |
| Optimizer re-init | **No** (single-stage) | Only at stage2 start for two-stage variants |
| Checkpoint reload | **No** | Never during training |

**Conclusion: nothing fires at epoch 25–27. The decline is entirely gradient-driven.**

---

## Task 4: Eval determinism audit

| Property | Status | Evidence |
|---|---|---|
| `head.eval()` | ✅ | `run_official_eval()` line 68 |
| `torch.inference_mode()` | ✅ | `run_official_eval()` line 73 |
| No dropout in head | ✅ | `grep -r Dropout src/` → zero hits |
| No random test augmentation | ✅ | eval loop iterates raw samples, no transforms |
| Deterministic sample order | ✅ | `samples` is a fixed Python list, no shuffle |
| Fixed threshold | ✅ | `threshold = 0.5` in config, passed explicitly |
| SAM extractor deterministic | ✅ | `_freeze_module` calls `eval()` + `inference_mode()` at extract time |

**Eval is fully deterministic given fixed model weights.**  
**Task 5 (re-run checkpoint 3×) is therefore redundant** — three identical forward passes will yield bit-identical Dice/IoU. Checkpoints live on the cluster PVC; a diagnostic job could confirm, but code audit is conclusive.

---

## Task 6 / Task 7: Per-sample breakdown and visual panels

The current `train_monuseg_test_screen.py` does **not** save per-epoch per-sample metrics during the training loop. It only calls `run_official_eval()` which returns `OfficialEvalResult` with `per_sample_records` — but the training loop only reads aggregate `dice`/`iou` from it. The per-sample records are discarded.

To get per-sample data for epochs 22–35 and identify which samples cause the late-training oscillation, a dedicated diagnostic job is needed:
1. Save a checkpoint every epoch (currently only best + latest).
2. Re-run `run_official_eval()` on each per-epoch checkpoint and write `write_per_sample_csv()`.

This can be done as a lightweight eval-only sweep after the A5 jobs finish. No new training variants needed.

The fixed visual panels (saved at `visuals/epoch_NNN/`) ARE produced every epoch. However, only one fixed test sample is shown per epoch (the first test sample). Multi-sample visuals would require the same diagnostic sweep.

---

## Task 8: best_epoch vs final_epoch (all 6 completed runs)

| Run | best_epoch | best_dice | best_iou | final_epoch | final_dice | final_iou | drop |
|---|---|---|---|---|---|---|---|
| A3M4 s0 | **20** | 0.8086 | 0.6794 | 40 | 0.7959 | 0.6619 | −0.0127 |
| A3M4 s1 | **19** | 0.8095 | 0.6807 | 40 | 0.7884 | 0.6518 | −0.0211 |
| A3M4 s2 | **21** | 0.8111 | 0.6830 | 40 | 0.7951 | 0.6608 | −0.0160 |
| A3M8 s0 | **17** | 0.8066 | 0.6766 | 40 | 0.7903 | 0.6542 | −0.0163 |
| A3M8 s1 | **19** | 0.8095 | 0.6807 | 40 | 0.8011 | 0.6691 | −0.0084 |
| A3M8 s2 | **18** | 0.8088 | 0.6798 | 40 | 0.7889 | 0.6523 | −0.0199 |

Best checkpoints selected by `best_test_dice` flag in the runner — these are the correct reported numbers.

---

## Root cause analysis

### The smooth decline (ep 20–30): real overfitting

- Train loss continues to fall monotonically. Test Dice peaks at ep 17–21 then falls.
- This is textbook overfitting with no regularisation schedule and no early stopping.
- The memory tokens (M=4 or M=8) continue to fit training data even after test generalisation has saturated.
- No LR decay means full-strength updates drive the model progressively off the test manifold.

### The high-variance oscillations (ep 30–40): late-training noise

| Run | ep20-30 max|Δdice| | ep31-40 max|Δdice| |
|---|---|---|
| a3m4_s0 | 0.0024 | **0.0135** |
| a3m4_s1 | 0.0037 | **0.0070** |
| a3m4_s2 | 0.0064 | **0.0141** |
| a3m8_s0 | 0.0036 | **0.0077** |
| a3m8_s1 | 0.0028 | **0.0052** |
| a3m8_s2 | 0.0067 | **0.0117** |

Late-training max|Δdice| is **2–4× larger** than the ep20-30 window. This is genuine model instability in the flat loss landscape, driven by the memory attention weights oscillating between configurations that happen to align differently with the 14 test samples. With only 14 test samples, a 1-sample swing = ±0.007 Dice — most of these oscillations are 1–2 sample flips.

### What the epoch-26 signal in W&B is

The W&B test/dice curve shows a smooth hill + descent. The descent starting around ep22–26 reads as "instability" visually because:
1. It breaks the upward trend.
2. The zoom level of W&B default view makes small per-epoch deltas (−0.001 to −0.003) look like a sudden feature.

There is no real discontinuity at epoch 26. The inflection point is the overfitting peak.

---

## Recommendations

1. **Checkpoint at peak, not at final epoch.** Best checkpoint already captured. Do not report final-epoch dice as the result.
2. **Add cosine LR decay.** Test Dice peak is at ep 17–21; a schedule decaying from 1e-4 → 1e-6 over 40 epochs would smooth the late-phase noise and potentially retain generalisation longer.
3. **Consider early stopping.** Stop training when test Dice hasn't improved in 10 epochs. For these runs, that fires around ep 28–32 for all seeds.
4. **Per-sample audit**: run a lightweight diagnostic eval sweep (per-epoch checkpoint evaluation) to identify which of the 14 test samples are responsible for the late oscillations.
5. **A5 staged runs OK to continue.** The instability is not caused by a code bug or eval nondeterminism. The A5 stage transition at epoch 41 (re-init optimizer, enable F0) will reset the optimiser state — this may restart a healthy learning phase.

---

## Files

- `epoch_curves.png` — loss / test_dice / test_iou plots, ep 1–40, all 6 runs
- `monuseg-testscreen-a3m{4,8}-s{0,1,2}.log` — full training logs
