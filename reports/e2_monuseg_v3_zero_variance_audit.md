# v3 Zero-Variance Audit

**Date:** 2026-05-27  
**Question:** Is 0.8228 ± 0.000 (v3, all_refinement_layers_memory_aug80) genuine stability, or a seed/config/checkpoint collision?

---

## W&B Run Identity

| Seed | W&B Run ID | Job Name | Status |
|------|------------|----------|--------|
| 0 | `1da2jppa` | monuseg-ts80-v3-allrefmem-s0 | Succeeded |
| 1 | `k85f2hqm` | monuseg-ts80-v3-allrefmem-s1 | Succeeded |
| 2 | `qi7q1j64` | monuseg-ts80-v3-allrefmem-s2 | Error (metrics saved) |

Run IDs are **distinct** — these are 3 separate W&B runs. ✓

---

## Seed Propagation

Seeds were passed as `--seed 0`, `--seed 1`, `--seed 2` via RunAI job submission commands (verified from `runai describe job` output). The training script calls `set_seed(seed)` which sets:
- `torch.manual_seed(seed)`
- `torch.cuda.manual_seed_all(seed)`
- `numpy.random.seed(seed)` (via `np.random.default_rng(seed)` for augmentation)
- `random.seed(seed)`

Seeds 0, 1, 2 are distinct integers → all random generators seeded differently. ✓

---

## Full-Precision Dice Values

Values from W&B summary (`final_epoch/dice`, 5 significant decimal places):

| Seed | Final Epoch Dice (5dp) |
|------|----------------------|
| 0 | 0.82294 |
| 1 | 0.82275 |
| 2 | 0.82290 |

**Computed statistics:**
- Mean: 0.82286
- Deviations: [+0.00008, −0.00011, +0.00004]
- Sample variance: 10.05 × 10⁻⁹
- Sample std: **0.0001002**

**Finding:** The reported ±0.000 is a rounding artifact. True std ≈ 1 × 10⁻⁴. This is genuinely very low variance — approximately 10× lower than v1 (std ≈ 0.001) and 40× lower than the batch 3 winner (std ≈ 0.004) — but not literally zero.

The correct reporting is **0.8229 ± 0.0001** (to 4dp) or **0.823 ± 0.000** (to 3dp, where rounding is responsible for "zero").

---

## Checkpoint SHA256 (truncated from W&B)

| Seed | SHA256 (first 20 chars) |
|------|------------------------|
| 0 | `b0c0c2027d6bf2064724...` |
| 1 | `717bd3fbf89c7c51be29...` |
| 2 | `426d6d2297d1c0e798df...` |

All prefixes are distinct → checkpoints are different files. ✓  
Full SHA256 verification requires PVC access at `/storage/nada/test_screen_80/v3_all_refine_mem/s{seed}/eval/test_metrics.json` (contains `checkpoint_final_sha256`).

---

## Oracle Dice (best_test_epoch)

All 3 seeds achieved **exactly 0.8232** (W&B summary `best_test/dice`):

| Seed | Oracle Dice | Best Epoch |
|------|------------|-----------|
| 0 | 0.82319 | 70 |
| 1 | 0.82318 | 70 |
| 2 | 0.82323 | 70 |

Oracle mean: 0.82320, std: 0.000022 (std at 5dp rounded to 0). Best epoch is consistently ep70 across all seeds.

---

## Possible Explanations for Low Variance

1. **Task is well-conditioned at this scale:** 14 test samples, binary segmentation with a converged architecture. At 80 epochs with aug + cosine LR, the model may be near a stable basin where seed differences in initialization/data-order average out.

2. **Memory attention acts as a regularizer:** The gated memory modules, by introducing a soft ensemble-like update at each level, may reduce sensitivity to random initialization noise.

3. **Augmentation smoothing:** autosam_dense_v1 with 37 training samples and varied augmentations may lead to near-identical loss landscapes across seeds by epoch 80.

4. **Not a collision:** Distinct run IDs, distinct seed values, distinct checkpoint SHA256 prefixes rule out trivial collision.

---

## Verdict

**Not a collision.** The ±0.000 appearance is rounding. True std ≈ 1e-4 (~10 Dice points in the 4th decimal), which is genuinely excellent reproducibility. v3 claims **0.8229 ± 0.0001** at final epoch.

---

## Action

Submit fresh seeds 3/4/5 (Block A) to confirm stability beyond the original 3 seeds.  
Script: `scripts/submit_monuseg_verification.py --block a`  
Expected: if std stays ≤ 0.001, v3 is confirmed as a stable architecture candidate.
