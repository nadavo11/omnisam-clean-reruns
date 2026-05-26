# Split Contamination Audit

Date: 2026-05-26

## Verdict Summary

- **RED**: MoNuSeg ablation conclusions are invalid. The historical selection metric came from the official training split, not a held-out validation or test split.
- **YELLOW**: Checkpoints and training curves are still useful as candidates and diagnostics, but not as evidence.
- **GREEN**: GlaS official metrics in `results/official_metrics/glas/*` are held-out test evaluations with provenance in the run manifests.

## What Failed

The MoNuSeg protocol was treated as an official evaluation path while the evidence stack was still anchored to the training split. That contaminates:

- model selection
- ablation ranking
- checkpoint choice for figures
- any claim that a variant "works" on MoNuSeg

## Evidence Table

| Run / Artifact | Checkpoint | Train split used? | Eval split actually used? | Dice / IoU | Visuals | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| `outputs/monuseg_A5lrn_staged_from_A3s0_e40` | `outputs/monuseg_A5lrn_staged_from_A3s0_e40/checkpoint.pt` | yes | train for selection / official headline; test exists in summary but was not the selection basis | train: 0.8539 / 0.7459; test: 0.8184 / 0.6934 | `outputs/monuseg_A5lrn_staged_from_A3s0_e40/visuals/` | yellow candidate, red evidence |
| `results/official_metrics/monuseg/A0/official_metrics.json` | `/storage/nada/outputs/fixed_resize_enhancement/monuseg_A0_fpn2_d128_512_e40/checkpoint.pt` | yes | train | 0.7862 / 0.6504 | none in repo | red |
| `results/official_metrics/monuseg/A2/official_metrics.json` | `/storage/nada/outputs/fixed_resize_enhancement/monuseg_A2_fpn2_fpn1refine_d128_512_e40/checkpoint.pt` | yes | train | 0.7750 / 0.6354 | none in repo | red |
| `results/official_metrics/monuseg/A3/official_metrics.json` | `/storage/nada/outputs/fixed_resize_enhancement/monuseg_A3_fpn2fpn1memoryattn_512_m8_e40/checkpoint.pt` | yes | train | 0.7117 / 0.5572 | none in repo | red |
| `results/official_metrics/monuseg/A5_all_at_once/official_metrics.json` | `/storage/nada/outputs/fixed_resize_enhancement/monuseg_A5lrn_fpn2fpn1f0lrn_512_s0_e40/checkpoint.pt` | yes | train | 0.7169 / 0.5639 | none in repo | red |
| `results/official_metrics/monuseg/final_staged_official_metrics.json` | `/storage/nada/outputs/fixed_resize_enhancement/monuseg_A5lrn_staged_from_A3s0_e40/checkpoint.pt` | yes | train | 0.8539 / 0.7459 | none in repo | red |
| `results/official_metrics/monuseg/freeze_A3_control_official_metrics.json` | `/storage/nada/outputs/fixed_resize_enhancement/monuseg_staged40_freezeA3_s0_e40/checkpoint.pt` | yes | train | 0.6777 / 0.5203 | none in repo | red |
| `results/official_metrics/glas/A0/official_metrics.json` | `/storage/nada/outputs/fixed_resize_enhancement/glas_A0_fpn2_d128_224_autosamaug_e200/checkpoint.pt` | no | test | 0.8792 / 0.7917 | none in repo | green |
| `results/official_metrics/glas/A2/official_metrics.json` | `/storage/nada/outputs/fixed_resize_enhancement/glas_A2_fpn2_fpn1refine_d128_224_autosamaug_e200/checkpoint.pt` | no | test | 0.8531 / 0.7519 | none in repo | green |
| `results/official_metrics/glas/A3/official_metrics.json` | `/storage/nada/outputs/fixed_resize_enhancement/glas_A3_m8_fpn2fpn1memoryattn_224_autosamaug_e200/checkpoint.pt` | no | test | 0.8496 / 0.7458 | none in repo | green |
| `results/official_metrics/glas/A5_all_at_once/official_metrics.json` | `/storage/nada/outputs/fixed_resize_enhancement/glas_A5lrn_fpn2fpn1f0lrn_224_autosamaug_e200/checkpoint.pt` | no | test | 0.8492 / 0.7456 | none in repo | green |
| `results/official_metrics/glas/final_staged_official_metrics.json` | `/storage/nada/outputs/fixed_resize_enhancement/glas_A5lrn_staged_from_A3s0_e40/checkpoint.pt` | no | test | 0.8438 / 0.7373 | none in repo | green |
| `results/official_metrics/glas/freeze_A3_control_official_metrics.json` | `/storage/nada/outputs/fixed_resize_enhancement/glas_staged40_freezeA3_s0_e40/checkpoint.pt` | no | test | 0.8338 / 0.7231 | none in repo | green |
| `paper_export/metrics/monuseg/*.json` | various source checkpoints | yes | train | see files | n/a | red historical conversion |
| `paper_export/metrics/glas/*.json` | various source checkpoints | yes | train | see files | n/a | red historical conversion |

## Notes

- The MoNuSeg final checkpoint still exists as a candidate. Its train-set headline metrics are not to be used for claims.
- The GlaS official metrics remain useful as held-out baselines while the MoNuSeg rerun stack is rebuilt.
- Any paper-facing table derived from train-evaluated MoNuSeg metrics is invalid until rerun under E1/E2 discipline.

---

## Event 2: First Clean E1 Matrix — Premature Test Exposure (2026-05-26)

**Severity:** ORANGE (premature official test eval; not train-on-test leakage)  
**Status:** Fixed; v2 reruns submitted

**What happened:**  
`train_clean_ablation.py` commit 946590a introduced `--run-test-eval` with `default=True`.  
The E1 launcher never passed `--no-run-test-eval`.  
All 9 E1 jobs (A0/A2/A3 × seeds 0/1/2) evaluated on `dataset["test"]` (14 official test crops)  
immediately after best-val checkpoint selection — before E1 model selection was complete.

**Secondary issue:** HF SAM-3 fallback returned 256ch for all FPN levels.  
A2/A3 ran with non-paper-equivalent channel layout (f1_channels=256 instead of 64).

**Remediation:**  
- `--run-test-eval` default → False  
- E1 split contract guard added (raises if run-test-eval + E1 job type without explicit override)  
- `--no-run-test-eval` added to launcher  
- Regression test: `tests/test_e1_e2_split_contract.py`  
- V2 reruns: `monuseg_E1_{A0,A2,A3}_s{0,1,2}_clean_v2`  
- Quarantined: `invalidated_train_eval/premature_test_exposure_first_clean_matrix.md`  
- Full audit: `e1_split_audit_after_first_matrix.md`

**Jobs affected:** monuseg_E1_A0_s{0,1,2}_clean, monuseg_E1_A2_s{0,1,2}_clean, monuseg_E1_A3_s{0,1,2}_clean

