# Checkpoint Index

All checkpoints at: `/storage/nada/outputs/fixed_resize_enhancement/` (run:AI cluster)

## MoNuSeg (512×512, 40 epochs, no augmentation, seed 0)

| Variant key | Source dir | Dice | IoU |
|---|---|---|---|
| a0_fpn2 | monuseg_A0_fpn2_d128_512_e40 | 0.8103 | 0.6835 |
| a2_fpn2_fpn1_refine | monuseg_A2_fpn2_fpn1refine_d128_512_e40 | 0.8423 | 0.7284 |
| a3_memory | monuseg_A3_fpn2fpn1memoryattn_512_m8_e40 | 0.8346 | 0.7171 |
| a5_all_at_once | monuseg_A5lrn_fpn2fpn1f0lrn_512_s0_e40 | 0.8392 | 0.7247 |
| final_staged | monuseg_A5lrn_staged_from_A3s0_e40 | 0.8539 | 0.7459 |
| freeze_a3_control | monuseg_staged40_freezeA3_s0_e40 | 0.8234 | 0.7010 |

## GlaS (224×224, 200 epochs, autosam_dense_v1 augmentation, seed 0)

| Variant key | Source dir | Dice | IoU |
|---|---|---|---|
| a0_fpn2 | glas_A0_fpn2_d128_224_autosamaug_e200 | 0.9374 | 0.8847 |
| a2_fpn2_fpn1_refine | glas_A2_fpn2_fpn1refine_d128_224_autosamaug_e200 | 0.9405 | 0.8897 |
| a3_memory | glas_A3_m8_fpn2fpn1memoryattn_224_autosamaug_e200 | 0.9243 | 0.8635 |
| a5_all_at_once | glas_A5lrn_fpn2fpn1f0lrn_224_autosamaug_e200 | 0.9259 | 0.8656 |

## Known Discrepancies

**MoNuSeg A5 all-at-once**: Locked value in `official_result_lock.yaml` is 0.8124 (from older
run `ce-monu-a5-v2-2930351` without fixed-resize protocol). The fixed-resize checkpoint above
gives 0.8392. The locked value is kept for historical reference; the reproducible value is 0.8392.
This affects only the A5 row in tables — our method (final_staged) is not affected.

## runai Jobs Used for Metric Extraction

- `mystorage` pod (persistent storage pod): used for `cat` of summary.json files
- `ckpt-probe-9591552`: submitted to inspect checkpoint state_dict key names (pending output)
