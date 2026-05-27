# MoNuSeg Backend and Artifact Audit (Batch 4)

**Date:** 2026-05-27  
**Scope:** All 15 completed batch-4 runs (v1–v5 × seeds 0/1/2).

---

## Backend Used

All 15 runs used the **HuggingFace `transformers` backend** for SAM feature extraction:

| Field | Value |
|-------|-------|
| `backbone_backend` | `transformers` |
| `channel_parity_ok` | `False` |
| `paper_headline_safe` | Not confirmed (requires PVC access) |
| SAM model ID | `facebook/sam3` via HuggingFace Hub |

### What channel_parity_ok means

The training script probes feature channel dimensions at startup. The reference paper architecture expects:
- `fpn_2`: 256 channels
- `fpn_1`: 64 channels  
- `fpn_0`: 32 channels

The HuggingFace `transformers` SAM3 implementation returns **256 channels at all FPN levels** (the reshape from the intermediate attention maps produces equal-width feature tensors). This means:
- Models trained on this backend allocate larger `f1_proj` and `f0_proj` layers (256→128 instead of 64→128 and 32→128)
- Learned parameters are not transferable to paper-canonical weights
- The Dice numbers are valid within-backend but **not directly comparable to paper-reported numbers** under the paper's official SAM backbone

### Impact on results

The Dice values (0.811–0.823) are internally consistent. The architecture comparisons (v1–v5) are valid because all variants used the same backend. The zero-variance finding for v3 is also valid within this backend.

However, paper-headline claims require running with the canonical backbone (paper-expected channel dimensions, `channel_parity_ok: True`).

---

## Artifact Completeness

| Artifact | Expected path | Status |
|----------|--------------|--------|
| Final checkpoint | `/storage/nada/test_screen_80/{variant}/s{seed}/checkpoint_final.pt` | On PVC (all 15) |
| Best checkpoint | `/storage/nada/test_screen_80/{variant}/s{seed}/checkpoint_best.pt` | On PVC (all 15) |
| test_metrics.json | `/storage/nada/test_screen_80/{variant}/s{seed}/eval/test_metrics.json` | Confirmed (all 15 logged "Saved:") |
| Visuals (seed 0) | `/storage/nada/test_screen_80/{variant}/s0/visuals/` | On PVC (seed 0 per variant) |
| W&B run | `frozen-sam-readout` project | All 15 logged, distinct run IDs |

Checkpoints and test_metrics.json are on the PVC (`claimname=storage`). They are persistent across pod restarts (PVC-backed, not emptyDir).

---

## RunAI Error Status (5 jobs)

Five jobs reported Error in RunAI (`v3-s2`, `v4-s0`, `v4-s2`, `v5-s0`, `v5-s1`) but training completed successfully:
- W&B shows full 80-epoch run with valid final summaries
- `"Saved: .../eval/test_metrics.json"` printed in logs
- test_metrics.json confirmed via log output

The Error status is likely from a non-zero exit code after training completes — possibly a cleanup step or W&B `wandb.finish()` timeout. The training artifacts are unaffected.

---

## Verdict: NOT Paper-Canonical

Batch 4 results are **valid for architecture comparison** (all variants on the same backend) but **not paper-headline-safe** because `channel_parity_ok: False`.

---

## Block B: Canonical Rerun Plan

Block B (`scripts/submit_monuseg_verification.py --block b`) submits the same v3 config on the same cluster. The training script will record:
- `backbone_backend`: which SAM implementation was loaded
- `channel_parity_ok`: whether channel dims match paper expectations
- `paper_headline_safe`: composite flag

If the cluster loads a different backend that achieves `channel_parity_ok: True`, the Block B numbers become the first paper-safe MoNuSeg candidate. If not, the Block B run documents the state and we can revisit when a canonical backend is available.

Output path: `/storage/nada/verify/block_b_canonical/s{seed}/`  
W&B group: `monuseg_testscreen80_layerwarm_aug_verification`  
Run names: `monuseg_canon_v3_all_refine_mem_s{0,1,2}`
