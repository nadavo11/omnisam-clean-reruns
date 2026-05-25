# paper_export — Staged Multiscale Frozen-SAM3 Readout

All paper-ready assets for submission. Generated 2026-05-25 from commit `9591552`.

## Directory Layout

```
paper_export/
  tables/          — CSV + LaTeX booktabs for all 5 paper tables
  figures/
    method/        — architecture diagrams (TBD — add manually)
    qualitative/   — sample visual panels from make_official_visuals.py
    plots/         — metric comparison plots (TBD)
  metrics/         — official_metrics.json per dataset/variant
    monuseg/{a0_fpn2, a2_fpn2_fpn1_refine, a3_memory, a5_all_at_once,
             final_staged, freeze_a3_control}/
    glas/{a0_fpn2, a2_fpn2_fpn1_refine, a3_memory, a5_all_at_once}/
  reports/         — migration and validation reports
  latex_snippets/  — ready-to-\input LaTeX fragments
  configs/         — official eval protocol YAML configs
  commands/        — shell scripts to reproduce training + eval
  source_refs/     — git commit, checkpoint index, run:AI job log
```

## Key Numbers (headline metrics, seed 0)

| Dataset | Method | Dice | IoU |
|---|---|---|---|
| MoNuSeg | **Ours (staged)** | **0.8539** | **0.7459** |
| MoNuSeg | A3 (Stage 1 only) | 0.8346 | 0.7171 |
| MoNuSeg | A0 (anchor) | 0.8103 | 0.6835 |
| GlaS | A2 (best) | 0.9405 | 0.8897 |
| GlaS | A0 (anchor) | 0.9374 | 0.8847 |

## Pending Before Submission

- [ ] Update A5 lock value (0.8124 → 0.8392) after decision
- [ ] Add qualitative figure panels from `outputs/official_visuals/`
- [ ] Fill in GPU-hours in `latex_snippets/reproducibility_statement.tex`
- [ ] Add method diagram to `figures/method/`
- [ ] Per-sample CSV for GlaS (re-eval needed on cluster)
- [ ] Checkpoint key remapper for migrated model loading
