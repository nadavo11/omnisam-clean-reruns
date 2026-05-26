# E1 Split Audit — First Clean Matrix

**Audit date:** 2026-05-26  
**Auditor:** automated (Claude Code)  
**Branch:** codex/clean-e1-runai-wandb  
**Audit commits:** ec2f540 (channel probe) · 946590a (test eval) · 272cde4 (SAM3 HF fix)

---

## Summary Verdict

**ALL 9 runs: ORANGE_PREMATURE_TEST**

The training script `train_clean_ablation.py` contained `--run-test-eval` with `default=True`.  
The launcher `launch_clean_e1.py` did not pass `--no-run-test-eval`.  
All 9 E1 jobs therefore evaluated on `iter_monuseg_binary_samples(split="test")` — the official  
14-sample HuggingFace test split — immediately after selecting the best validation checkpoint.

The **val metrics** (`eval_metrics_val.json`) remain valid for E1 model selection.  
The **test metrics** (`eval_metrics_test.json`) are premature test exposure and must be quarantined.

---

## Run-by-Run Audit

| run_name | variant | seed | metric_file | eval_split_in_payload | sample_source | verdict |
|---|---|---:|---|---|---|---|
| monuseg_E1_A0_s0_clean | a0_fpn2 | 0 | eval_metrics_test.json | test | iter_monuseg_binary_samples(split="test") — 14 official test crops | 🟠 ORANGE_PREMATURE_TEST |
| monuseg_E1_A0_s1_clean | a0_fpn2 | 1 | eval_metrics_test.json | test | iter_monuseg_binary_samples(split="test") — 14 official test crops | 🟠 ORANGE_PREMATURE_TEST |
| monuseg_E1_A0_s2_clean | a0_fpn2 | 2 | eval_metrics_test.json | test | iter_monuseg_binary_samples(split="test") — 14 official test crops | 🟠 ORANGE_PREMATURE_TEST |
| monuseg_E1_A2_s0_clean | a2_fpn2_fpn1_refine | 0 | eval_metrics_test.json | test | iter_monuseg_binary_samples(split="test") — 14 official test crops | 🟠 ORANGE_PREMATURE_TEST |
| monuseg_E1_A2_s1_clean | a2_fpn2_fpn1_refine | 1 | eval_metrics_test.json | test | iter_monuseg_binary_samples(split="test") — 14 official test crops | 🟠 ORANGE_PREMATURE_TEST |
| monuseg_E1_A2_s2_clean | a2_fpn2_fpn1_refine | 2 | eval_metrics_test.json | test | iter_monuseg_binary_samples(split="test") — 14 official test crops | 🟠 ORANGE_PREMATURE_TEST |
| monuseg_E1_A3_s0_clean | a3_memory | 0 | eval_metrics_test.json | test | iter_monuseg_binary_samples(split="test") — 14 official test crops | 🟠 ORANGE_PREMATURE_TEST |
| monuseg_E1_A3_s1_clean | a3_memory | 1 | eval_metrics_test.json | test | iter_monuseg_binary_samples(split="test") — 14 official test crops | 🟠 ORANGE_PREMATURE_TEST |
| monuseg_E1_A3_s2_clean | a3_memory | 2 | eval_metrics_test.json | test | iter_monuseg_binary_samples(split="test") — 14 official test crops | 🟠 ORANGE_PREMATURE_TEST |

---

## Split Manifest Evidence

**Validation IDs** (from `configs/clean_reruns/monuseg_fixed_sample_ids.json`):
```
TCGA-38-6178-01Z-00-DX1
TCGA-HE-7129-01Z-00-DX1
TCGA-A7-A13E-01Z-00-DX1
TCGA-HE-7128-01Z-00-DX1
TCGA-G9-6356-01Z-00-DX1
TCGA-AY-A8YK-01A-01-TS1
```
(6 samples from the official training set; selection_policy: first_6_official_train_ids_in_source_order)

**Official test set** (from HuggingFace dataset["test"]):  
14 separate crops not present in the training set. These are the crops used in the premature test eval.

**Code path that caused the violation:**
```python
# train_clean_ablation.py lines 244-245
train_samples = list(iter_monuseg_binary_samples(split="train"))
test_samples  = list(iter_monuseg_binary_samples(split="test"))   # ← official test
# ...
# line 212 (old): --run-test-eval default=True
# line 536-565: if args.run_test_eval: runs eval on test_samples
```

---

## Val Metrics (Valid for E1 Selection)

These are uncontaminated and usable for E1 checkpoint/variant selection.

### A0 — a0_fpn2

| Seed | Best Val Dice | Best Val IoU | Best Epoch |
|---:|---:|---:|---:|
| 0 | 0.6997 | 0.5435 | 6 |
| 1 | 0.7007 | 0.5447 | 5 |
| 2 | 0.7019 | 0.5462 | 5 |
| **mean** | **0.7008** | **0.5448** | — |

### A2 — a2_fpn2_fpn1_refine

| Seed | Best Val Dice | Best Val IoU | Best Epoch |
|---:|---:|---:|---:|
| 0 | 0.7810 | 0.6418 | 20 |
| 1 | 0.7789 | 0.6390 | 37 |
| 2 | 0.7752 | 0.6343 | 35 |
| **mean** | **0.7784** | **0.6384** | — |

### A3 — a3_memory

| Seed | Best Val Dice | Best Val IoU | Best Epoch |
|---:|---:|---:|---:|
| 0 | 0.7818 | 0.6428 | 39 |
| 1 | 0.7798 | 0.6404 | 34 |
| 2 | 0.7862 | 0.6489 | 34 |
| **mean** | **0.7826** | **0.6440** | — |

**Note on FPN channel parity:** HF SAM-3 fallback returned 256ch for all FPN levels.
A2/A3 heads ran with f1_channels=256 (paper design: 64). Results show correct trend
(A0 < A2 < A3) but magnitudes are not paper-equivalent. See `fpn_channel_parity_audit.md`.

---

## Quarantined Test Metrics (Reconnaissance Only — Do Not Use for Selection)

| run_name | Test Dice | Test IoU | Contamination Risk |
|---|---:|---:|---|
| A0 s0 | 0.7390 | 0.5869 | ORANGE — not to be used for E1 selection or E2 gating |
| A0 s1 | 0.7381 | 0.5858 | ORANGE |
| A0 s2 | 0.7401 | 0.5883 | ORANGE |
| A2 s0 | 0.7999 | 0.6674 | ORANGE |
| A2 s1 | 0.7984 | 0.6654 | ORANGE |
| A2 s2 | 0.7996 | 0.6669 | ORANGE |
| A3 s0 | 0.8011 | 0.6691 | ORANGE |
| A3 s1 | 0.8055 | 0.6752 | ORANGE |
| A3 s2 | 0.8043 | 0.6735 | ORANGE |

These numbers may be seen as private reconnaissance. They must not be used to:
- Select the E1 champion variant
- Report paper results
- Gate E2 (official test) evaluation
- Inform hyperparameter tuning based on test signal

---

## Root Cause

`train_clean_ablation.py` commit 946590a introduced `--run-test-eval` with `default=True`.
The launcher `launch_clean_e1.py` and submit script never passed `--no-run-test-eval`.

---

## Fixes Applied

| File | Change |
|---|---|
| `scripts/train_clean_ablation.py` | `--run-test-eval` default changed to `False` |
| `scripts/train_clean_ablation.py` | E1 split contract guard: raises if `--run-test-eval` + E1 job type without `--allow-premature-test-eval` |
| `scripts/train_clean_ablation.py` | `feature_shapes.json` logged at startup (backbone backend, channel counts, parity) |
| `launcher_repo/launch_clean_e1.py` | Added `--no-run-test-eval` to training command |
| `scripts/submit_clean_e1_runai.py` | Added `--suffix` support for v2 runs; PVC paths include suffix |
| `tests/test_e1_e2_split_contract.py` | New: enforces all split contract rules |

---

## Action Required

1. ✅ Quarantine test metrics from first matrix (see `invalidated_train_eval/`)
2. ✅ Fix runner (done above)
3. ✅ Add regression test (done above)
4. ⏳ Rerun v2 matrix with `--suffix v2` — val only, no test eval
5. ⏳ After v2 completes: select E1 champion based on val metrics
6. ⏳ Only then: run official E2 test with locked candidate manifest
