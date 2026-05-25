# Extending omniSAM

## Adding a new dataset

1. Add a module under `src/frozen_sam_readout/data/` exposing
   `iter_<name>_binary_samples(*, split, limit, start_index, ...)` that yields
   dataclasses with `image`, `texture_a_mask` (foreground), `texture_b_mask`
   (background), and `crop_name` fields.
2. Register the iterator in `src/frozen_sam_readout/data/registry.py` under
   `DATASET_REGISTRY`.
3. Add a YAML config under `configs/official/` or `configs/research/`
   describing the resize policy and the eval threshold.

## Adding a new head variant

1. Add a class under `src/frozen_sam_readout/models/heads/` deriving from
   `nn.Module`, taking `**channels` kwargs and a `pyramid` dict at forward.
2. Add a block-level dependency under `src/frozen_sam_readout/models/blocks/`
   if needed.
3. Register the head in `src/frozen_sam_readout/models/registry.py` under
   `MODEL_REGISTRY`.
4. Add a shape-check test to `tests/test_feature_shapes.py`.

## Adding a new ablation

1. Add a YAML config under `configs/research/` (NOT `configs/official/`).
2. Run with `python scripts/eval_official.py --allow-nonofficial ...` so the
   `paper_headline_safe` flag is set to `false`.
3. Document the new column in `paper_assets/tables/` by extending the row
   ordering in `scripts/make_paper_tables.py`.

## Adding a new SAM backbone

Edit `frozen_sam_readout/sam/feature_extractor.py`:
- Add a new branch to `resolve_frozen_sam_backbone_family`.
- Add a new extractor class with `extract_sam_pyramid(image)` returning
  `((H, W), {"fpn_2": ..., "fpn_1": ..., "fpn_0": ...})`.
- Make sure all backbone parameters are frozen (`_freeze_module`).
