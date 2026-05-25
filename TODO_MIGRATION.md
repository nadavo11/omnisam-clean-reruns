# TODO — Migration

Tracking work that was intentionally deferred from the initial migration.

## Open items

- [ ] Wire the SAM-3 extractor to the actual SAM-3 vision-encoder API used in
      the source repo's `Sam3CoarseVsFineScaleFeatureExtractor`. The current
      `_Sam3PyramidExtractor` uses a generic `transformers.AutoModel` path
      that may need adjustment depending on the official SAM-3 release.
- [ ] Add a feature-cache producer (`scripts/build_feature_cache.py`) so
      stage-1 / stage-2 training does not re-run the frozen SAM encoder every
      epoch. The streaming loop in `train_stage{1,2}.py` is correct but slow.
- [ ] Port the data-augmentation policy `autosam_dense_v1` from the source
      repo (`src/rwtd_sam3/eval/frozen_mask_head_augmentations.py`).
- [ ] Port the boundary-auxiliary loss path referenced in
      `configs/research/boundary_auxiliary.yaml`.
- [ ] Multi-seed driver script that runs 3 seeds and aggregates the
      `direct_foreground_dice ± std` headline. Currently each `eval_official.py`
      call records one seed; tables read the lock file mean.
- [ ] Add a per-sample CSV dump alongside `official_metrics.json` to enable
      bootstrap CIs in paper-table generation.
- [ ] Confirm exact SAM-2 channel counts (`f2_channels=256`, `f1_channels=64`,
      `f0_channels=32`) at runtime — `_DEFAULT_CHANNELS` is the standard
      `hiera_l` shape but should be discovered from the live extractor.
- [ ] CI: add a GitHub Actions workflow running `pytest` and the smoke script
      on every push.

## Closed items

- [x] Move package to `frozen-sam-readout` with `src/` layout.
- [x] Verbatim port of `compute_binary_metrics` + `BinaryMetrics`.
- [x] Official-protocol guards.
- [x] Stage-1 + Stage-2 training scripts with `alpha = sigmoid(log(0.05/0.95))`
      init and zero-init final 1×1.
- [x] `official_metrics.json` schema lock.
- [x] Paper tables + comparison-grid visuals.
