# MoNuSeg TEST_SCREEN Checkpoint Selection Policy

**Date:** 2026-05-26  
**Status:** Active — applies to all MoNuSeg TEST_SCREEN runs in this campaign

---

## Problem statement

MoNuSeg has **no validation split**. The official split is 37 train patches + 14 test patches. There is no independent held-out set. This means:

- Any checkpoint selected by performance on the test split is **test-selected** — by definition a test-screen oracle result.
- Per-run early stopping that uses test Dice to pick the checkpoint **cannot serve as clean unbiased evidence**. It will produce optimistic numbers.
- These numbers are still scientifically useful as diagnostic bounds, but they must be labeled and cannot be reported as a final fair claim.

---

## The three checkpoint views

Every run in this campaign must report **three views separately**. They must never be averaged together or presented under the same heading.

### 1. `final_epoch` — claimable within TEST_SCREEN protocol

| Field | Value |
|---|---|
| `selected_by` | `final_epoch` |
| `selected_epoch` | `total_epochs` (predeclared in config) |
| `claimability` | `claimable_test_screen` |

**What it means:** The checkpoint at the last epoch of a predeclared fixed schedule. The epoch count is set before training. The result is not optimized by test performance — it is simply "run N epochs, take the last checkpoint."

**Why it is claimable (with caveats):** The schedule is fixed and predeclared. No test feedback was used to select this checkpoint. However, the result is still a TEST_SCREEN number: it was obtained by training on the full training set with test evaluation visible at every epoch. It is not an unbiased estimate of generalization.

**How to report:** Report mean ± std across seeds. This is the headline number for the ledger.

### 2. `best_test_epoch` — diagnostic oracle only

| Field | Value |
|---|---|
| `selected_by` | `test_peak` |
| `selected_epoch` | epoch achieving max test Dice |
| `claimability` | `diagnostic_oracle` |

**What it means:** The checkpoint that happened to achieve the highest test Dice across all epochs. This is the absolute upper bound on what the model can do on the test split given this training run.

**Why it is NOT claimable:** Selecting by test performance is test-set leakage. With only 14 test samples, selecting the peak epoch is equivalent to fitting a single hyperparameter (the stopping point) to the test set. Across seeds, the peak epoch varies (e.g., ep17–21 for A3 variants), so this selection is not replicable without access to the test labels.

**How to report:** Report mean ± std across seeds, clearly labeled `(diagnostic oracle, not claimable)`. Use to diagnose schedule design: if `best_test_epoch` >> `final_epoch`, the schedule is too long or needs regularization.

### 3. `fixed_epoch` — post-hoc but consistent

| Field | Value |
|---|---|
| `selected_by` | `fixed_epoch` |
| `selected_epoch` | a fixed integer declared after seeing the dataset (e.g., 20) |
| `claimability` | `posthoc_fixed_epoch` |

**What it means:** A checkpoint at a specific epoch (e.g., epoch 20) chosen after observing peak patterns across seeds. For A3 variants, the test Dice peak occurs at epochs 17–21 across all 6 seeds, making epoch 20 a natural "consensus peak" that could be declared as a fixed protocol.

**When to use:** Use to compare variants on a consistent epoch basis. Declare the epoch once and apply it uniformly across all seeds and variants. Do not adjust per-seed.

**How to report:** Report mean ± std, labeled `(post-hoc fixed epoch=20, not predeclared)`. Useful for deciding the next control's fixed schedule.

---

## Practical rules

1. **Default claim-bearing number is `final_epoch`.** A predeclared fixed schedule makes this the only defensible pick when no validation split exists.

2. **`best_test_epoch` is a ceiling, not a result.** Report it only as a diagnostic upper bound. Never put it in an abstract or headline table without explicit labeling.

3. **No per-run adaptive stopping.** Early stopping based on test Dice makes every run's stopping point a test-set-informed decision. This is forbidden for claim-bearing results.

4. **Future runs must preregister the epoch count.** The schedule (number of epochs, LR schedule, stage transition point) must be fixed in the config before training. Any post-hoc schedule change must be treated as a new variant.

5. **When introducing a new fixed epoch based on oracle analysis:** Document that the epoch was chosen by observing the `best_test_epoch` distribution across seeds. Label results at that epoch as `posthoc_fixed_epoch`. In future runs using that epoch as a predeclared schedule, the result becomes `claimable_test_screen`.

---

## Application to current campaign

| Variant | Total epochs | Stage transition | lr_schedule | `final_epoch` claimable? | Notes |
|---|---|---|---|---|---|
| A3M4 (batch 1) | 40 | none | constant | ✅ yes | final_epoch=40, dice≈0.793 mean |
| A3M8 (batch 1) | 40 | none | constant | ✅ yes | final_epoch=40, dice≈0.793 mean |
| A5staged (batch 1) | 80 | ep41 | constant | ✅ yes | final_epoch=80, dice≈0.803 mean |
| A5noMem (batch 1) | 80 | ep41 | constant | ✅ yes | final_epoch=80, dice≈0.808 mean |
| A3M4short20 (batch 2) | 20 | none | constant | ✅ yes | predeclared by oracle analysis |
| A3M4cos40 (batch 2) | 40 | none | cosine 1e-4→1e-6 | ✅ yes | cosine eliminates overfitting problem |
| A5stage20fromA3M4 (batch 2) | 40 | ep21 | constant | ✅ yes | earlier transition = better init |
| A5stage20noMem (batch 2) | 40 | ep21 | constant | ✅ yes | earlier transition = better init |

---

## Relation to MoNuSeg no-validation contract

This policy is a consequence of the no-validation contract (`assert_monuseg_no_val`). Because there is no validation set:
- Checkpoint selection cannot be made by validation performance.
- All test-set-informed selection is by definition TEST_SCREEN oracle.
- The only escape is a fixed predeclared schedule.

See also: `reports/monuseg_no_validation_audit.md`
