# Official Eval Candidates

These are the checkpoints worth rerunning under the clean E1/E2 discipline.

## Yellow candidates

| Candidate | Path | Why it stays in play |
| --- | --- | --- |
| MoNuSeg final staged checkpoint | `outputs/monuseg_A5lrn_staged_from_A3s0_e40/checkpoint.pt` | best available checkpoint for rebuilding, but prior MoNuSeg headline metrics are train-contaminated |

## Green baselines to keep

| Baseline | Path | Status |
| --- | --- | --- |
| GlaS A0 official metrics | `results/official_metrics/glas/A0/official_metrics.json` | green held-out test eval |
| GlaS A2 official metrics | `results/official_metrics/glas/A2/official_metrics.json` | green held-out test eval |
| GlaS A3 official metrics | `results/official_metrics/glas/A3/official_metrics.json` | green held-out test eval |
| GlaS A5 official metrics | `results/official_metrics/glas/A5_all_at_once/official_metrics.json` | green held-out test eval |
| GlaS final staged official metrics | `results/official_metrics/glas/final_staged_official_metrics.json` | green held-out test eval |
| GlaS freeze-A3 control official metrics | `results/official_metrics/glas/freeze_A3_control_official_metrics.json` | green held-out test eval |

## Launch order

1. Smoke / fit test with a tiny train-only run.
2. Validation ablations for `A0`, `A2`, and `A3`.
3. Official test reruns for the winners only.

## Required launch discipline

- fixed split manifests
- explicit `eval_split` in every metric payload
- fixed visual sample IDs
- W&B logging for train, validation, and qualitative artifacts

