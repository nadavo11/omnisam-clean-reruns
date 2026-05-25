# Official Protocols

These two protocols define the only paper-headline-safe evaluation views.
Anything that violates a guard listed below is research-only.

## MoNuSeg-strict-512 (`configs/official/monuseg_strict_512.yaml`)

- Dataset: `RationAI/MoNuSeg` HF mirror, official challenge split (30 train +
  14 test). Excludes the 7 extra HF `tissue == 0` train rows.
- Pre-SAM resize: 512 x 512 bilinear.
- SAM: frozen, no prompt encoder, no mask decoder.
- Training: 40 ep stage-1 → 40 ep stage-2, AdamW, lr 1e-4, wd 1e-4, bs 1.
- Eval: `direct_foreground_dice`, `direct_foreground_iou`, threshold 0.5.
- Headline metric source: `frozen_sam_readout/evaluation/metrics.py::compute_binary_metrics`.

## GlaS-fixed-224 (`configs/official/glas_fixed_224.yaml`)

- Dataset: local Warwick GlaS layout (AutoSAM flat directory).
- Pre-SAM resize: 224 x 224 bilinear.
- SAM: frozen, no prompt encoder, no mask decoder.
- Training: 200 ep stage-1 → 40 ep stage-2, AdamW, lr 1e-4, wd 1e-4, bs 1.
- Eval: `direct_foreground_dice`, `direct_foreground_iou`, threshold 0.5.

## Hard guards (enforced by `frozen_sam_readout.utils.config.validate_official_guards`)

- `sam.frozen` must be `true`.
- `sam.use_prompt_encoder` must be `false`.
- `sam.use_mask_decoder` must be `false`.
- `evaluation.threshold` must equal `0.5`.

The audit script `scripts/audit_protocol.py` runs these guards and can also
load the backbone and run `assert_sam_frozen` against it.

## Headline metric contract

The `official_metrics.json` schema written by `scripts/eval_official.py`:

```json
{
  "dataset": "monuseg",
  "protocol": "monuseg_strict_512",
  "variant": "final_staged",
  "checkpoint": "...",
  "seed": 0,
  "headline_metrics": {"direct_foreground_dice": 0.0, "direct_foreground_iou": 0.0},
  "auxiliary_metrics": {"upstream_eval_dice": null, "upstream_eval_iou": null},
  "threshold": 0.5,
  "metric_source": "frozen_sam_readout/evaluation/metrics.py::compute_binary_metrics",
  "prediction_resolution": "...",
  "gt_resolution": "...",
  "paper_headline_safe": true
}
```

`paper_headline_safe` flips to `false` if and only if the eval was launched
with `--allow-nonofficial`.
