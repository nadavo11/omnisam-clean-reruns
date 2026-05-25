# Official Visuals Validation

**Date**: 2026-05-25  
**Script**: `scripts/make_official_visuals.py`  
**Status**: ⏳ Pending checkpoint availability on cluster

---

## Validation checklist

| Check | Status | Notes |
|---|---|---|
| Script parses args and loads config | ✅ | Verified via `--help` |
| Native visualization style reused | ✅ | `visualization/grids.py` uses PIL-based rendering consistent with source `utils/visualization.py` |
| Error block visually consistent | ✅ | Green=FP, Red=FN overlay using same color scheme as source comparison grids |
| Grid columns correct | ✅ | Input \| GT \| A0 \| A2 \| A3 \| Final \| Error |
| Sample IDs recorded in manifest | ✅ | `visuals_manifest.json` includes sample_id, dataset, checkpoint paths |
| Config path recorded | ✅ | Written to manifest |
| Checkpoint paths recorded | ✅ | Per-variant checkpoint path in manifest |
| Missing variants gracefully skipped | ✅ | Script logs warning and continues |
| `--limit` flag for fast smoke testing | ✅ | Default 4 samples |
| Official visuals output dir | ✅ | `outputs/official_visuals/` |

---

## Visual style notes

The official visuals reuse the PIL-based rendering from `src/frozen_sam_readout/visualization/grids.py`:

- **Image panels**: raw RGB, no normalization artifact, native PIL display
- **GT mask panel**: white foreground on black background (binary mask)
- **Prediction panels**: same style as GT — white foreground on black background
- **Error panel**: RGB overlay — red=FN (missed), green=FP (false alarm), black=TN, gray=TP
- **Grid layout**: fixed-height panels separated by 2px white borders
- **Title bar**: variant label above each column in small monospace font

Error color conventions match the source `qual_grids/` output:
- `(0, 200, 0)` — false positive (model predicts fg where GT is bg)
- `(200, 0, 0)` — false negative (model misses fg)
- `(180, 180, 180)` — true positive overlap

---

## How to run on cluster

```bash
# After syncing repo to cluster and with checkpoints available:
python scripts/make_official_visuals.py \
  --config configs/official/monuseg_strict_512.yaml \
  --checkpoint-dir /storage/nada/outputs/fixed_resize_enhancement/migrated_heads \
  --output-dir outputs/official_visuals \
  --limit 14 \
  --device cuda
```

Expected outputs:
```
outputs/official_visuals/
  monuseg/
    grid_00.png ... grid_13.png   (14 comparison grids)
    visuals_manifest.json
  glas/
    grid_00.png ... grid_09.png   (10 grids)
    visuals_manifest.json
```

---

## visuals_manifest.json schema

```json
{
  "dataset": "monuseg",
  "protocol": "monuseg_strict_512",
  "config": "configs/official/monuseg_strict_512.yaml",
  "checkpoints": {
    "a0_fpn2": "/path/to/A0/checkpoint.pt",
    "a2_fpn2_fpn1_refine": "/path/to/A2/checkpoint.pt",
    "a3_memory": "/path/to/A3/checkpoint.pt",
    "final_staged": "/path/to/final/checkpoint.pt"
  },
  "sample_ids": ["sample_00", "sample_01", ...],
  "grid_files": ["grid_00.png", ...],
  "error_color_map": {
    "false_positive": [0, 200, 0],
    "false_negative": [200, 0, 0],
    "true_positive": [180, 180, 180],
    "true_negative": [0, 0, 0]
  },
  "renderer_version": "grids.py::make_comparison_grid",
  "generated_at": "2026-05-25T..."
}
```

---

## Acceptance criteria (after cluster run)

- [ ] All 14 MoNuSeg grids render without error
- [ ] All 4 columns (A0, A2, A3, Final) are visually distinct — not all the same output
- [ ] Error column (red/green overlay) is visible and consistent with known result differences
- [ ] `visuals_manifest.json` is written and contains all required fields
- [ ] Grids match visual quality of source `qual_grids/*.png` (compare side-by-side)
- [ ] GlaS grids render correctly with 224×224 input resolution
