# Migration Report

## Source

`/home/nada/PycharmProjects/texture representations/` — research repo containing
many SAM-1/2/3 evaluation tracks. The frozen-feature staged readout lives
inside `src/rwtd_sam3/models/sam3_frozen_multiscale_mask_head.py` (3561 LoC)
together with several legacy variants we did NOT port.

## What was migrated

| Source | Target |
|---|---|
| `rwtd_sam3/models/frozen_sam_pyramid_extractor.py` | `frozen_sam_readout/sam/feature_extractor.py` (cleaned, frozen-enforced) |
| `rwtd_sam3/models/sam3_frozen_multiscale_mask_head.py` (selective) | `frozen_sam_readout/models/heads/*.py` + `blocks/*.py` (clean reimplementation following METHOD.md) |
| `rwtd_sam3/data/monuseg_binary.py` | `frozen_sam_readout/data/monuseg.py` |
| `rwtd_sam3/data/glas_binary.py` | `frozen_sam_readout/data/glas.py` |
| `rwtd_sam3/eval/metrics.py::compute_binary_metrics` and `BinaryMetrics` | `frozen_sam_readout/evaluation/metrics.py` (verbatim copy) |
| `rwtd_sam3/utils/visualization.py` (column layout) | `frozen_sam_readout/visualization/` |
| `experiments/corrected_global_memory_feature_readout/phase4_phase5_summary.md` | `results/official_result_lock.yaml` |

## What was NOT migrated (intentionally)

- All non-frozen-feature tracks (`autosam_*`, `sam2_baseline`,
  `architexture_binary`, `cstd_binary`, `detexture_*`, `rwtd_*` automatic-mask
  variants).
- The `*_failure_aware`, `*_multibank_null`, `*_serialized_progressive`,
  `*_local_routed` head variants from the source — those are old ablations.
- The boundary-aux head, deep-DPM/HDBSCAN feature clusterers, and the
  self-SAM SSL pilot.
- The legacy CLI surface (`rwtd_sam3.cli`) is replaced by a small set of
  task-specific scripts under `scripts/`.

## Design choices

- **Clean reimplementation of heads** instead of literal copy. The source
  file mixes >10 variants with shared state and many runtime branches; a
  clean re-implementation per `METHOD.md` is easier to reason about,
  audit-friendly, and matches the documented spec exactly (memory tokens M=8,
  `D=128`, `alpha = sigmoid(log(0.05/0.95))`, zero-init final 1×1 conv).
- **Verbatim metric port** because `compute_binary_metrics` is the official
  headline scoring path. Any drift here would invalidate the locked numbers.
- **Hard guards on official configs** via `validate_official_guards`, with an
  explicit `--allow-nonofficial` opt-out and a `paper_headline_safe` flag in
  `official_metrics.json`.
- **Streaming feature extraction** in the training scripts. A real run wants
  a cached-feature dataset — tracked in `TODO_MIGRATION.md`.

## Verification

- Tests: `pytest -q` runs 5 suites (sam_frozen, metric_equivalence,
  official_config_guards, feature_shapes, smoke). All shape and metric tests
  should pass on a fresh checkout without GPU.
- Audit: `python scripts/audit_protocol.py --config configs/official/monuseg_strict_512.yaml`
  validates the YAML guards.
- Numerical reproducibility of the locked headline numbers requires a SAM-3
  checkpoint and the actual training pipeline; see `TODO_MIGRATION.md`.

## Open risks

- SAM-3 extractor: stub uses a generic `AutoModel.vision_encoder` call that
  may need adjustment for the actual checkpoint API.
- Default channel counts are pinned to standard SAM-2 (`256/64/32`); confirm
  against the live extractor when wiring up SAM-3.
