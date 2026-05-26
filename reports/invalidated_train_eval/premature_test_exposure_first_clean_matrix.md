# QUARANTINED: Premature Test Exposure — First Clean E1 Matrix

**Status:** ORANGE_PREMATURE_TEST — results are for private reconnaissance only  
**Date quarantined:** 2026-05-26  
**Reason:** E1 jobs evaluated on official test IDs before E1 model selection was complete

Do NOT use these test numbers to:
- Select E1 champion
- Report paper results
- Gate E2 (official test) evaluation
- Tune hyperparameters

See `../e1_split_audit_after_first_matrix.md` for the full audit.

## Quarantined Test Numbers (Private Reconnaissance Only)

| Variant | s0 Test Dice | s1 Test Dice | s2 Test Dice | Mean Test Dice |
|---|---:|---:|---:|---:|
| A0 (fpn_2 only) | 0.7390 | 0.7381 | 0.7401 | 0.7391 |
| A2 (+ fpn_1 refine) | 0.7999 | 0.7984 | 0.7996 | 0.7993 |
| A3 (+ memory attn) | 0.8011 | 0.8055 | 0.8043 | 0.8036 |

These numbers are also non-paper-equivalent because the HF SAM-3 fallback returned
256 channels for all FPN levels (paper design: fpn_2=256, fpn_1=64, fpn_0=32).
See `../fpn_channel_parity_audit.md`.
