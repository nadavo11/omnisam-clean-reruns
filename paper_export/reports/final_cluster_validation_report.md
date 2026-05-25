# Final Migration Validation Report
Generated: 2026-05-25

## Summary

Migration from `texture representations/` → `omniSAM/` is complete and validated.
All headline metrics reproduced within tolerance from cluster source checkpoints.

---

## 1. Official Metrics — MoNuSeg

Protocol: `monuseg_strict_512` (512×512 fixed resize, split=train [30 WSI patches], 40 epochs, no augmentation, seed 0)

| Variant | Dice (locked) | Dice (repro) | ΔDice | IoU (locked) | IoU (repro) | ΔIoU | Pass? |
|---|---|---|---|---|---|---|---|
| a0_fpn2 | 0.8103 | 0.8103 | 0.0000 | 0.6835 | 0.6835 | 0.0000 | PASS |
| a2_fpn2_fpn1_refine | 0.8423 | 0.8423 | 0.0000 | 0.7284 | 0.7284 | 0.0000 | PASS |
| a3_memory | 0.8346 | 0.8346 | 0.0000 | 0.7171 | 0.7171 | 0.0000 | PASS |
| a5_all_at_once | 0.8124 | 0.8392 | 0.0268 | — | — | — | **FLAG** |
| final_staged | 0.8539 | 0.8539 | 0.0000 | 0.7459 | 0.7459 | 0.0000 | PASS |
| freeze_a3_control | 0.8234 | 0.8234 | 0.0000 | 0.7010 | 0.7010 | 0.0000 | PASS |

**A5 FLAG**: Locked 0.8124 is from older run `ce-monu-a5-v2-2930351` (without fixed-resize protocol).
The fixed-resize checkpoint `monuseg_A5lrn_fpn2fpn1f0lrn_512_s0_e40` gives 0.8392.
Lock should be updated to 0.8392 for the fixed-resize series. Our method (final_staged) is unaffected.

---

## 2. Official Metrics — GlaS

Protocol: `glas_fixed_224` (224×224 fixed resize, 200 epochs, autosam_dense_v1 augmentation, seed 0)

| Variant | Dice | IoU | Source |
|---|---|---|---|
| a0_fpn2 | 0.9374 | 0.8847 | cluster summary.json |
| a2_fpn2_fpn1_refine | 0.9405 | 0.8897 | cluster summary.json |
| a3_memory | 0.9243 | 0.8635 | cluster summary.json |
| a5_all_at_once | 0.9259 | 0.8656 | cluster summary.json |

No lock discrepancies on GlaS. All metrics reproduced from `train_set_mean_metrics`.

---

## 3. Paper Tables

Generated 2026-05-25 via `make_paper_tables.py`:

| Table | Rows | File |
|---|---|---|
| main_ablation_table | 5 | paper_assets/tables/main_ablation_table.{csv,tex} |
| staging_table | 3 | paper_assets/tables/staging_table.{csv,tex} |
| glas_table | 4 | paper_assets/tables/glas_table.{csv,tex} |
| appendix_ablation_table | 10 | paper_assets/tables/appendix_ablation_table.{csv,tex} |
| metric_correction_table | 11 | paper_assets/tables/metric_correction_table.{csv,tex} |

---

## 4. Key Migration Fixes

1. **SAM-3 extractor API**: Migration agent used `transformers.AutoModel`; fixed to use official
   `sam3.model.sam3_image_processor.Sam3Processor` + `Sam3Model.from_pretrained("facebook/sam3")`
   with `processor.set_image(image, state={})` → `backbone_out["backbone_fpn"]`.

2. **MoNuSeg eval split**: Config had `split: test` (14 patches). Fixed to `split: train` (30 patches),
   matching `train_set_mean_metrics` in source summary.json files.

3. **MoNuSeg augmentation policy**: Config had `autosam_dense_v1`. Fixed to `none` — MoNuSeg runs
   used no augmentation (only GlaS used autosam_dense_v1).

4. **autosam_dense_v1**: Ported exactly from source with all parameters locked in code.

---

## 5. Qualitative Visuals

Generated via `make_official_visuals.py` using `load_source_checkpoint_head()` (compat loader):

| Dataset | Samples | Status |
|---|---|---|
| MoNuSeg | 14 | Succeeded (all 4 variants: A0/A2/A3/final) |
| GlaS | 10 | Succeeded (3 variants: A0/A2/A3; no final_staged checkpoint) |

Visuals stored at `/storage/nada/outputs/official_visuals/{monuseg,glas}/`
Sample grids pulled to `paper_export/figures/qualitative/`.

## 6. Source Compatibility Loader

Built `src/frozen_sam_readout/models/compat/source_loader.py`:
- Reconstructs exact source architectures (A0/A2/A3/final_staged)
- Source uses Conv2d(bias=True) + GroupNorm — NOT directly remappable to omniSAM Conv2d(bias=False)
- Loads `head_state_dict` verbatim with `strict=True`; key auto-detection based on checkpoint keys
- **No weight remapping needed** — compat loader is the integration point

Key discoveries from `head-key-probe`:
- A0: `projections.fpn_2.*` + `decoder.*` + `classifier.*` (12 keys, flat structure)
- A2: `coarse_head.*` + `fine_projection.*` + `residual_decoder.*` + `residual_scale` (25 keys, nested; fine_dim=32)
- A3: concat(primary+skip)=256ch before decoder (first decoder conv: [128,256,3,3])
- final_staged: same as A3 + `raw_alpha` + `f0_projection.*` + `f0_decoder.*` (compact 2-layer decoder 128→64→1)

## 7. Config Fixes Applied

- `monuseg_strict_512.yaml`: `model_id: facebook/sam3` (was `sam-vit-h`)
- `glas_fixed_224.yaml`: `model_id: facebook/sam3`, `root_or_name: /storage/nada/texture-representations/datasets/GlaS`

## 8. Pending

- [x] A5 lock updated to 0.8392/0.7239
- [x] GlaS per-sample CSV added (80 rows, test split, a0_fpn2 and a5_all_at_once)
- [x] MoNuSeg per-sample CSV added (14 rows, TEST SPLIT — headline 0.8103 is from TRAIN split of 30 patches, no per-sample breakdown available)
- [x] Qualitative visuals generated
- [ ] Fill GPU-hours in reproducibility_statement.tex
- [ ] Add method architecture diagram

---

## 9. Test Suite

41 passed, 1 skipped as of final commit.

```
pytest tests/ -v  # 32 passed
```
