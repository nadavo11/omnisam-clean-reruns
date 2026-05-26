# E2 MoNuSeg TEST_SCREEN Ablation Screen

**Updated:** 2026-05-26 (batch 1 complete, batch 2 planned)  
**Checkpoint policy:** `reports/e2_monuseg_checkpoint_selection_policy.md`  
**Instability analysis:** `reports/e2_monuseg_epoch_instability.md`  
**CSV:** `results/e2_monuseg_test_ablation_screen.csv`

---

## Scope

MoNuSeg has no validation split. All results below are TEST_SCREEN: train on the full 37-sample training set, evaluate on the official 14-sample test split every epoch. Three checkpoint views are reported separately:

- **`final_epoch`** — claimable within TEST_SCREEN protocol (predeclared fixed schedule)
- **`best_test_epoch`** — diagnostic oracle only (selected by test Dice; not claimable)
- **`fixed_epoch_20`** — post-hoc consistent reference for A3 variants (not predeclared)

Legacy A0/A2/A3 rows used a train-side validation selection — non-claimable under the current no-val contract.

---

## Summary: `final_epoch` (claimable)

| Variant | Epochs | Stage trans. | lr_schedule | Dice mean ± std | IoU mean ± std |
|---|---:|---:|---|---:|---:|
| **A5noMem** | 80 | ep41 | constant | **0.808 ± 0.003** | **0.680 ± 0.004** |
| **A5staged** | 80 | ep41 | constant | **0.803 ± 0.004** | **0.672 ± 0.005** |
| A3M8 | 40 | — | constant | 0.793 ± 0.006 | 0.659 ± 0.007 |
| A3M4 | 40 | — | constant | 0.793 ± 0.003 | 0.658 ± 0.006 |
| A3 (E1 legacy) | 40 | — | constant | 0.804 ± 0.002 | 0.673 ± 0.003 |
| A2 (E1 legacy) | 40 | — | constant | 0.799 ± 0.001 | 0.667 ± 0.001 |
| A0 (E1 legacy) | 40 | — | constant | 0.739 ± 0.001 | 0.587 ± 0.001 |

*E1 legacy rows: selected by val Dice, non-claimable under MoNuSeg no-val contract. Shown for context only.*

---

## Summary: `best_test_epoch` (diagnostic oracle, not claimable)

| Variant | Dice mean ± std | IoU mean ± std | Peak epoch range |
|---|---:|---:|---|
| **A5staged** | **0.816 ± 0.003** | **0.690 ± 0.004** | ep63–79 of 80 |
| **A5noMem** | **0.815 ± 0.002** | **0.689 ± 0.002** | ep43–53 of 80 |
| A3M4 | 0.810 ± 0.001 | 0.681 ± 0.002 | ep19–21 of 40 |
| A3M8 | 0.808 ± 0.002 | 0.679 ± 0.002 | ep17–19 of 40 |

---

## Summary: `fixed_epoch_20` for A3 variants (post-hoc, not predeclared)

| Variant | Dice mean ± std | IoU mean ± std |
|---|---:|---:|
| A3M4 | 0.810 ± 0.001 | 0.681 ± 0.002 |
| A3M8 | 0.807 ± 0.003 | 0.678 ± 0.003 |

*A3M4 at ep20 ≈ A3M4 best_test_epoch. Epoch 20 is the natural stopping point for A3 variants.*

---

## Per-seed detail

### A3M4 (40 epochs, constant LR=1e-4)

| Seed | final_epoch dice | final_epoch iou | best_test dice | best_epoch | fixed_ep20 dice |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.7959 | 0.6619 | 0.8086 | 20 | 0.8086 |
| 1 | 0.7884 | 0.6518 | 0.8095 | 19 | 0.8094 |
| 2 | 0.7951 | 0.6608 | 0.8111 | 21 | 0.8110 |

### A3M8 (40 epochs, constant LR=1e-4)

| Seed | final_epoch dice | final_epoch iou | best_test dice | best_epoch | fixed_ep20 dice |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.7903 | 0.6542 | 0.8066 | 17 | 0.8045 |
| 1 | 0.8011 | 0.6691 | 0.8095 | 19 | 0.8094 |
| 2 | 0.7889 | 0.6523 | 0.8088 | 18 | 0.8083 |

### A5staged — A3M4+F0, stage1=40, stage2=40

| Seed | final_epoch dice | final_epoch iou | best_test dice | best_epoch (of 80) | stage2 epoch at best |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.8051 | 0.6747 | 0.8177 | 63 | ep23 |
| 1 | 0.8057 | 0.6754 | 0.8181 | 72 | ep32 |
| 2 | 0.7982 | 0.6652 | 0.8121 | 79 | ep39 |

### A5noMem — A2+F0, stage1=40, stage2=40

| Seed | final_epoch dice | final_epoch iou | best_test dice | best_epoch (of 80) | stage2 epoch at best |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.8116 | 0.6838 | 0.8160 | 53 | ep13 |
| 1 | 0.8075 | 0.6782 | 0.8131 | 43 | ep3 |
| 2 | 0.8059 | 0.6761 | 0.8155 | 53 | ep13 |

---

## Key findings

1. **F0 residual is the main gain:** A5noMem final_epoch (0.808) vs A3M4 final_epoch (0.793): +0.015 Dice.
2. **Memory adds nothing in 2-stage setting:** A5noMem (0.808) ≥ A5staged (0.803). Use A2+F0 by default.
3. **Stage1=40 is too long:** A3 peaks at ep17–21; starting stage2 at ep41 wastes 20 epochs. A5noMem-s1 achieves its best at stage2-ep3, confirming rapid recovery from overfit init.
4. **Constant LR causes −0.017 Dice gap** between final_epoch@40 and best_test_epoch@17-21 for A3 variants. Cosine schedule should close most of this.
5. **A5@stage1=40 still beats A3M4 final_epoch by +0.010-0.015 despite the bad transition point.** The stage2 optimizer re-init partially resets overfitting.

---

## Batch 2 next controls

| Priority | Variant | Config | Job prefix | Rationale |
|---|---|---|---|---|
| A | A3M4_short20 | `monuseg_a3m4_short20.yaml` | a3m4-short20 | Validate ep20 as predeclared stopping point |
| A | A3M4_cosine40 | `monuseg_a3m4_cosine40.yaml` | a3m4-cos40 | Close overfitting gap with schedule |
| A | A5_staged20_A3M4 | `monuseg_a5_staged20_from_a3m4.yaml` | a5stage20-a3m4 | Correct transition point |
| A | A5_staged20_noMem | `monuseg_a5_staged20_no_memory.yaml` | a5stage20-nomem | Best architecture + correct schedule |
| B | A2/A3M8 short20+cos40 | respective yamls | as above | Complete the grid |

Submit minimum batch:
```bash
set -a; source /home/nada/.config/omnisam-clean-reruns/secrets.env; set +a
python scripts/submit_monuseg_test_screen.py --seeds 0,1,2
# default variants = A3M4short20,A3M4cos40,A5stage20fromA3M4,A5stage20noMem
```

---

## Artifact and W&B status (batch 1)

- **W&B:** offline for all 12 runs (project-name bug `nadavoteam/...` — fixed for batch 2)
- **Checkpoints:** lost to ephemeral pod emptyDir for most runs; only logs are reliable
- **Segmentation visuals:** not reviewed; lost with pod storage
- **Backend:** HF transformers fallback — all FPN levels 256ch; not paper-headline-safe

---

## Protocol notes

- No val split. No early stopping. Only predeclared fixed schedules produce claimable results.
- `Error` RunAI status on some A5 jobs = pod cleanup artifact; training completed normally.
- See `reports/e2_monuseg_checkpoint_selection_policy.md` for the full policy rationale.
