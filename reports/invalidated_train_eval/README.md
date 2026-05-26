# Invalidated Train-Eval Artifacts

These files are quarantined because they were used as evidence for MoNuSeg
ablation claims even though the selection/evaluation discipline was tied to the
training split.

Do not use them for paper claims.

## Quarantined families

- `results/official_metrics/monuseg/*`
- `paper_export/metrics/monuseg/*`
- `paper_export/metrics/glas/*` source-conversion files from the old train-set
  summary pipeline
- `outputs/monuseg_A5lrn_staged_from_A3s0_e40/*` as a candidate checkpoint and
  train-set diagnostic bundle

## Safe usage

- Checkpoints: candidate only
- Train curves: diagnosis only
- Visuals selected from train performance: diagnosis only
- Public claims: none

