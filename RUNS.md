# Runs

Table of official runs that contributed to `results/official_result_lock.yaml`.
Numbers are pulled from the source repo's
`experiments/corrected_global_memory_feature_readout/phase4_phase5_summary.md`.

| Dataset | Variant | Seed(s) | Dice | IoU | Source phase |
|---|---|---|---|---|---|
| MoNuSeg-strict-512 | a0_fpn2 | 0 | 0.8103 | 0.6835 | phase4 |
| MoNuSeg-strict-512 | a2_fpn2_fpn1_refine | 0 | 0.8423 | 0.7284 | phase4 |
| MoNuSeg-strict-512 | a3_memory | 0 | 0.8339 | 0.7161 | phase4 |
| MoNuSeg-strict-512 | a5_all_at_once | 0 | 0.8124 | 0.6865 | phase5 |
| MoNuSeg-strict-512 | final_staged | 0,1,2 (mean) | 0.8545 ± 0.0050 | 0.7467 ± 0.0073 | phase5 |
| MoNuSeg-strict-512 | freeze_a3_control | 0 | 0.8234 | 0.7010 | phase5 |
| GlaS-fixed-224 | a0_fpn2 | 0 | 0.9374 | 0.8847 | phase4 |
| GlaS-fixed-224 | a2_fpn2_fpn1_refine | 0 | 0.9405 | 0.8897 | phase4 |
| GlaS-fixed-224 | a3_memory | 0 | 0.9243 | 0.8635 | phase4 |
| GlaS-fixed-224 | a5_safe | 0 | 0.9259 | 0.8656 | phase5 |

New runs should be appended here as `official_metrics.json` files land under
`results/official_metrics/<dataset>/<run_name>/`.
