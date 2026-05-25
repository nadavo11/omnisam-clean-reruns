# Results

All numbers below are pinned in `results/official_result_lock.yaml`.

## MoNuSeg-strict-512

| Variant | Dice | IoU | Notes |
|---|---|---|---|
| A0 (`fpn_2` only) | 0.8103 | 0.6835 | Coarse baseline |
| A2 (`fpn_2`+`fpn_1` refine) | 0.8423 | 0.7284 | +residual refine |
| A3 (A2 + memory attn, M=8) | 0.8339 | 0.7161 | Stage-1 output |
| A5 all-at-once | 0.8124 | 0.6865 | Joint training, no warm-start |
| **Final staged (mean of 3 seeds)** | **0.8545 ± 0.0050** | **0.7467 ± 0.0073** | Stage-1 → Stage-2 + F0 |
| Freeze-A3 control | 0.8234 | 0.7010 | Stage-2 with A3 frozen |

The staged method strictly dominates the all-at-once variant and the
freeze-A3 control, validating both (a) the warm-started F0 residual and (b)
keeping A3 trainable in stage-2.

## GlaS-fixed-224

| Variant | Dice | IoU | Notes |
|---|---|---|---|
| A0 | 0.9374 | 0.8847 | |
| **A2 (best)** | **0.9405** | **0.8897** | |
| A3 | 0.9243 | 0.8635 | |
| A5-safe | 0.9259 | 0.8656 | |

On GlaS the A2 head is the strongest variant on the locked headline metrics;
the staged F0 stage and memory attention do not improve over A2 at the
fixed-224 protocol. See `TODO_MIGRATION.md` for follow-up items.

## Provenance

Numbers come from
`experiments/corrected_global_memory_feature_readout/phase4_phase5_summary.md`
in the source repo (`/home/nada/PycharmProjects/texture representations/`).
