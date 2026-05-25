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

- [x] A5 lock updated to 0.8392/0.7239 (fixed-resize checkpoint); old 0.8124 audited in metric_correction_table
- [x] Qualitative QC grids: 3 MoNuSeg + 3 GlaS pulled to `figures/qualitative/`; manifests + figure_grid.tex included
- [ ] Fill in GPU-hours in `latex_snippets/reproducibility_statement.tex`
- [ ] Add method diagram to `figures/method/`
- [x] GlaS per-sample CSV: a0_fpn2 (80 rows) + a5_all_at_once (80 rows) in `metrics/glas/`
- [x] Checkpoint compat loader: `src/frozen_sam_readout/models/compat/` loads A0/A2/A3/final_staged from source checkpoints
