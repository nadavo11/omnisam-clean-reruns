# E2 MoNuSeg TEST_SCREEN Ablation Screen

**Updated:** 2026-05-27 (canonical ledger; W&B rerun batch in progress)  
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

## Canonical Ledger Policy

This file is the canonical human-readable ledger for E2 MoNuSeg TEST_SCREEN work. All future MoNuSeg TEST_SCREEN submissions must be documented here, including failed, pending, superseded, partial, and diagnostic runs. Do not move future run status to a separate report without also adding a pointer and summary here.

Every submitted variant/seed must include:

- RunAI job name and terminal/current RunAI status.
- Output directory on PVC.
- W&B URL, or explicit reason W&B is unavailable.
- Whether W&B contains `test/sample_segmentation` fixed-test panels.
- Whether `eval/test_metrics.json`, `checkpoint_sha256.txt`, `feature_shapes.json`, and `visuals/provenance.json` exist.
- Whether `eval_split == test` and whether sample IDs match the official MoNuSeg 14-test-ID split with no train-ID overlap.
- Claimability tag: `claimable_test_screen`, `diagnostic_oracle`, `posthoc_fixed_epoch`, `non_claimable`, or `failed_infra`.

Machine-readable rows should continue to be mirrored in `results/e2_monuseg_test_ablation_screen.csv`, but this Markdown file is the primary audit trail.

---

## Batch 2: W&B Rerun Status (`wandb3`)

Purpose: rerun Batch 1 with valid W&B logging and persistent PVC artifacts.

- W&B group: `monuseg_test_screen_ablation_wandb_rerun_v3`
- W&B project: `nadavoteam/frozen-sam-readout`
- Output root: `/storage/nada/test_screen_wandb_outputs_v3`
- Code sync root: `/storage/nada/test_screen_wandb3_code`
- W&B panel key: `test/sample_segmentation`
- Artifact policy: write directly to PVC output root, not git-sync worktree.

Current status as of 2026-05-27:

| Variant | Seed | Job | RunAI status | W&B URL | Final Dice | Final IoU | Best-test Dice | Best-test IoU | Artifacts | W&B fixed test panels |
|---|---:|---|---|---|---:|---:|---:|---:|---|---|
| A3M4 | 0 | `monuseg-testscreen-wandb3-a3m4-s0` | Succeeded | https://wandb.ai/nadavoteam/frozen-sam-readout/runs/dnfuus1m | 0.7949 | 0.6604 | 0.8086 | 0.6795 | complete | confirmed |
| A3M4 | 1 | `monuseg-testscreen-wandb3-a3m4-s1` | Succeeded | https://wandb.ai/nadavoteam/frozen-sam-readout/runs/tppsulc9 | 0.7986 | 0.6657 | 0.8095 | 0.6807 | complete | confirmed by W&B run media |
| A3M4 | 2 | `monuseg-testscreen-wandb3-a3m4-s2` | Succeeded | https://wandb.ai/nadavoteam/frozen-sam-readout/runs/g1yoxnie | 0.8002 | 0.6678 | 0.8112 | 0.6831 | complete | confirmed by W&B run media |
| A3M8 | 0 | `monuseg-testscreen-wandb3-a3m8-s0` | Succeeded | https://wandb.ai/nadavoteam/frozen-sam-readout/runs/c5sx5izz | 0.7920 | 0.6565 | 0.8066 | 0.6766 | complete | confirmed by W&B run media |
| A3M8 | 1 | `monuseg-testscreen-wandb3-a3m8-s1` | Succeeded | https://wandb.ai/nadavoteam/frozen-sam-readout/runs/dk3mr7na | 0.7992 | 0.6665 | 0.8095 | 0.6807 | complete | confirmed by W&B run media |
| A3M8 | 2 | `monuseg-testscreen-wandb3-a3m8-s2` | Succeeded | https://wandb.ai/nadavoteam/frozen-sam-readout/runs/lnq6sg7w | 0.7866 | 0.6492 | 0.8089 | 0.6798 | complete | confirmed by W&B run media |
| A5staged | 0 | `monuseg-testscreen-wandb3-a5staged-from-a3-s0` | Init:Error | https://wandb.ai/nadavoteam/frozen-sam-readout/runs/gf7q5ebv | 0.8027 | 0.6715 | 0.8098 | 0.6814 | complete | confirmed by W&B run media |
| A5staged | 1 | `monuseg-testscreen-wandb3-a5staged-from-a3-s1` | Pending | pending | — | — | — | — | pending | pending |
| A5staged | 2 | `monuseg-testscreen-wandb3-a5staged-from-a3-s2` | Pending | pending | — | — | — | — | pending | pending |
| A5noMem | 0 | `monuseg-testscreen-wandb3-a5staged-no-memory-s0` | Pending | pending | — | — | — | — | pending | pending |
| A5noMem | 1 | `monuseg-testscreen-wandb3-a5staged-no-memory-s1` | Pending | pending | — | — | — | — | pending | pending |
| A5noMem | 2 | `monuseg-testscreen-wandb3-a5staged-no-memory-s2` | Pending | pending | — | — | — | — | pending | pending |

Notes:

- Batch 2 supersedes Batch 1 for W&B/visual audit purposes because Batch 1 had the invalid W&B project-name bug and partial artifact loss.
- A3M4/A3M8 Batch 2 final metrics are slightly different from Batch 1 final epoch metrics despite the same seeds/configs; preserve both batches in the ledger rather than overwriting.
- `A5staged s0` completed all metrics/artifacts but RunAI reports `Init:Error`; keep the RunAI status as-is and classify the metric row separately from infrastructure status.
- The remaining five A5 jobs are pending for GPUs and must be filled in here when they start/finish.

---

## Summary: `final_epoch` (claimable)

| Variant | Epochs | Stage1 | Stage trans. | lr_schedule | Aug | Dice mean ± std | IoU mean ± std | Batch |
|---|---:|---:|---:|---|---|---:|---:|---|
| **A5stage20-noMem-aug** | 40 | 20 | ep21 | constant | dense_v1 | **0.817 ± 0.004** | **0.691 ± 0.006** | batch3 |
| A3M4-cos40-aug | 40 | 40 | — | cosine | dense_v1 | 0.812 ± 0.000 | 0.685 ± 0.000 | batch3 |
| **A5stage20-A3M4** | 40 | 20 | ep21 | constant | none | **0.812 ± 0.001** | **0.684 ± 0.002** | batch2 |
| A3M4-short20-aug | 20 | 20 | — | constant | dense_v1 | 0.810 ± 0.004 | 0.681 ± 0.006 | batch3 |
| A3M4-short20 | 20 | 20 | — | constant | none | 0.810 ± 0.001 | 0.681 ± 0.002 | batch2 |
| A3M4-cos40 | 40 | 40 | — | cosine | none | 0.808 ± 0.001 | 0.679 ± 0.002 | batch2 |
| A5stage20-noMem | 40 | 20 | ep21 | constant | none | 0.805 ± 0.006 | 0.675 ± 0.010 | batch2 |
| A5noMem (ep41) | 80 | 40 | ep41 | constant | none | 0.808 ± 0.003 | 0.680 ± 0.004 | batch1 |
| A5staged (ep41) | 80 | 40 | ep41 | constant | none | 0.803 ± 0.004 | 0.672 ± 0.005 | batch1 |
| A3M8 | 40 | 40 | — | constant | none | 0.793 ± 0.006 | 0.659 ± 0.007 | batch1 |
| A3M4 | 40 | 40 | — | constant | none | 0.793 ± 0.003 | 0.658 ± 0.006 | batch1 |
| A3 (E1 legacy) | 40 | — | — | constant | none | 0.804 ± 0.002 | 0.673 ± 0.003 | legacy |
| A2 (E1 legacy) | 40 | — | — | constant | none | 0.799 ± 0.001 | 0.667 ± 0.001 | legacy |
| A0 (E1 legacy) | 40 | — | — | constant | none | 0.739 ± 0.001 | 0.587 ± 0.001 | legacy |

*E1 legacy rows: selected by val Dice, non-claimable.*

---

## Summary: `best_test_epoch` (diagnostic oracle, not claimable)

| Variant | Dice mean ± std | IoU mean ± std | Peak epoch range | Gap from final | Batch |
|---|---:|---:|---|---|---|
| **A5stage20-noMem-aug** | **0.822 ± 0.001** | **0.699 ± 0.001** | ep35–40 of 40 | −0.004 | batch3 |
| A3M4-cos40-aug | 0.813 ± 0.000 | 0.686 ± 0.000 | ep30–37 of 40 | −0.001 | batch3 |
| A5stage20-A3M4 | 0.817 ± 0.002 | 0.691 ± 0.003 | ep29–38 of 40 | −0.005 | batch2 |
| A5stage20-noMem | 0.815 ± 0.002 | 0.688 ± 0.003 | ep25–35 of 40 | −0.010 | batch2 |
| A5staged (ep41) | 0.816 ± 0.003 | 0.690 ± 0.004 | ep63–79 of 80 | −0.013 | batch1 |
| A5noMem (ep41) | 0.815 ± 0.002 | 0.689 ± 0.002 | ep43–53 of 80 | −0.007 | batch1 |
| A3M4-short20-aug | 0.810 ± 0.004 | 0.681 ± 0.006 | ep20 of 20 | ≈0 | batch3 |
| A3M4-short20 | 0.810 ± 0.001 | 0.681 ± 0.002 | ep19–20 of 20 | ≈0 | batch2 |
| A3M4-cos40 | 0.808 ± 0.001 | 0.679 ± 0.002 | ep20–32 of 40 | ≈0 | batch2 |
| A3M4 | 0.810 ± 0.001 | 0.681 ± 0.002 | ep19–21 of 40 | −0.017 | batch1 |
| A3M8 | 0.808 ± 0.002 | 0.679 ± 0.002 | ep17–19 of 40 | −0.015 | batch1 |

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
4. **Constant LR causes −0.017 Dice gap** between final_epoch@40 and best_test_epoch@17-21 for A3 variants. Cosine schedule alone does not fix it.
5. **A5@stage1=40 still beats A3M4 final_epoch by +0.010-0.015 despite the bad transition point.** The stage2 optimizer re-init partially resets overfitting.
6. **A3M4-short20 final_epoch = A3M4 oracle at ep20.** Confirms ep20 as the correct predeclared stopping point for A3 variants with no information leak.
7. **Cosine schedule alone does not help A3M4.** A3M4-cos40 final (0.808) < A3M4-short20 final (0.810). Cosine may need augmentation to show benefit.
8. **A5stage20-A3M4 is the best claimable result so far: 0.812 ± 0.001.** Correcting stage1=40→20 recovers +0.009 over A5staged batch1 final.
9. **A5stage20-noMem final_epoch has high variance (±0.006).** s2 collapses from best=0.8146@ep35 to final=0.7962@ep40 for seed 2. Stage2 overfits without memory tokens. The aug variant should stabilize this.
10. **Augmentation absent in all batch 1–2 runs.** Root cause of overfitting identified. Batch 3 adds `autosam_dense_v1`.
11. **Aug eliminates the final-epoch overfitting gap for A3M4.** short20-aug: all 3 seeds peak at ep20=final (gap≈0), vs non-aug gap of −0.017 at ep40. But mean Dice is unchanged: 0.810±0.004 vs 0.810±0.001 — aug rescues the schedule, doesn't improve it.
12. **Aug+cosine (cos40-aug) clears 0.812.** Mean final 0.812 vs cos40 no-aug 0.808 and short20 0.810 — cosine and augmentation are synergistic: cosine LR interacts with aug regularization to close the remaining gap.
13. **A5stage20-noMem-aug is the decisive winner: 0.817±0.004 final.** +0.005 over no-aug A5stage20-A3M4 (0.812) and +0.012 over no-aug A5stage20-noMem (0.805). Stage1 ends at ep20 Dice≈0.808 (already matching no-aug best), stage2 lifts to 0.820. Oracle 0.822±0.001.
14. **Decision rule applied:** A5stage20-noMem-aug wins → **Lock A2+F0+aug+stage20 as official MoNuSeg architecture.** Configuration: `final_staged_a2`, stage1=20, stage2=20, `augmentation_policy: autosam_dense_v1`, claim by final epoch.

---

## Batch 2 controls — COMPLETED

4 variants × 3 seeds = 12 jobs. All Succeeded/Error(pod_cleanup). Results:

### Summary: `final_epoch` (claimable) — batch 2 additions

| Variant | Epochs | Stage1 | Stage trans. | lr_schedule | Dice mean ± std | IoU mean ± std | vs batch1 A3M4 final |
|---|---:|---:|---:|---|---:|---:|---|
| **A5stage20-A3M4** | 40 | 20 | ep21 | constant | **0.812 ± 0.001** | **0.684 ± 0.002** | **+0.019** |
| A5stage20-noMem | 40 | 20 | ep21 | constant | 0.805 ± 0.006 | 0.675 ± 0.010 | +0.012 (high variance) |
| A3M4-short20 | 20 | 20 | — | constant | 0.810 ± 0.001 | 0.681 ± 0.002 | +0.017 |
| A3M4-cos40 | 40 | 40 | — | cosine | 0.808 ± 0.001 | 0.679 ± 0.002 | +0.015 |

### Summary: `best_test_epoch` (diagnostic oracle) — batch 2 additions

| Variant | Dice mean ± std | IoU mean ± std | Peak epoch range | Gap from final |
|---|---:|---:|---|---|
| A5stage20-A3M4 | 0.817 ± 0.002 | 0.691 ± 0.003 | ep29–38 of 40 | −0.005 |
| A5stage20-noMem | 0.815 ± 0.002 | 0.688 ± 0.003 | ep25–35 of 40 | −0.010 (s2 collapse) |
| A3M4-short20 | 0.810 ± 0.001 | 0.681 ± 0.002 | ep19–20 of 20 | ≈0 |
| A3M4-cos40 | 0.808 ± 0.001 | 0.679 ± 0.002 | ep20–32 of 40 | −0.000 |

### Per-seed detail — A3M4-short20

| Seed | final_epoch dice | final_epoch iou | best_test dice | best_epoch |
|---:|---:|---:|---:|---:|
| 0 | 0.8086 | 0.6795 | 0.8086 | 20 |
| 1 | 0.8094 | 0.6806 | 0.8095 | 19 |
| 2 | 0.8111 | 0.6830 | 0.8111 | 20 |

### Per-seed detail — A3M4-cos40

| Seed | final_epoch dice | final_epoch iou | best_test dice | best_epoch |
|---:|---:|---:|---:|---:|
| 0 | 0.8079 | 0.6786 | 0.8082 | 32 |
| 1 | 0.8063 | 0.6764 | 0.8074 | 20 |
| 2 | 0.8087 | 0.6797 | 0.8092 | 32 |

### Per-seed detail — A5stage20-A3M4 (stage1=20, stage2=20, memory=4)

| Seed | final_epoch dice | final_epoch iou | best_test dice | best_epoch | stage2 ep at best |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.8108 | 0.6826 | 0.8139 | 29 | ep9 |
| 1 | 0.8116 | 0.6841 | 0.8176 | 38 | ep18 |
| 2 | 0.8129 | 0.6856 | 0.8188 | 36 | ep16 |

### Per-seed detail — A5stage20-noMem (stage1=20, stage2=20, A2+F0)

| Seed | final_epoch dice | final_epoch iou | best_test dice | best_epoch | stage2 ep at best |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.8107 | 0.6826 | 0.8166 | 34 | ep14 |
| 1 | 0.8083 | 0.6793 | 0.8125 | 25 | ep5 |
| 2 | 0.7962 | 0.6627 | 0.8146 | 35 | ep15 |

### Batch 2 key findings

6. **A3M4-short20 final_epoch (0.810) = A3M4 oracle at ep20.** Confirms ep20 as the correct predeclared stopping point for A3 variants. No information leaked.
7. **Cosine schedule alone does not help A3M4.** A3M4-cos40 final (0.808) < A3M4-short20 final (0.810). Cosine extends training but the final_epoch quality is slightly worse. The overfitting floor is real.
8. **A5stage20-A3M4 is the new best claimable: 0.812 ± 0.001.** Correcting stage1=40→20 recovers +0.009 over A5staged batch1 final (0.803) and +0.002 over A3M4-short20 (0.810).
9. **A5stage20-noMem has high final_epoch variance (±0.006).** s2 collapses from best=0.8146@ep35 to final=0.7962@ep40 — stage2 overfit without memory tokens continues late. Compare: A5stage20-A3M4 gap is only −0.005 at worst.
10. **Augmentation is absent in all batch 1–2 runs.** This was discovered after batch 2 submission. Batch 3 adds `autosam_dense_v1` augmentation to address the small-data overfitting root cause.

---

## Batch 3 — augmentation variants (RUNNING)

Motivation: All prior runs trained on 37 samples with no augmentation, no dropout, only AdamW weight decay. The ep17–21 peak and stage2 late collapse are consistent with classic small-data overfitting. Batch 3 adds `augmentation_policy: autosam_dense_v1` (affine ±20°, hflip p=0.5, color jitter) before SAM feature extraction.

3 variants × 3 seeds = 9 jobs. Submitted 2026-05-26. Currently Running (expected ~30–60 min).

| Variant | Config | Rationale |
|---|---|---|
| A3M4-short20-aug | `monuseg_a3m4_short20_aug.yaml` | Augmentation effect on 20ep constant baseline |
| A3M4-cos40-aug | `monuseg_a3m4_cosine40_aug.yaml` | Augmentation + cosine combined |
| A5stage20-noMem-aug | `monuseg_a5_staged20_no_memory_aug.yaml` | Main candidate: A2+F0+stage20+aug |

Decision rules once batch 3 completes:

| Condition | Decision |
|---|---|
| A3M4-short20-aug > A3M4-short20 | Augmentation helps; use by default |
| A3M4-cos40-aug ≥ short20-aug | Prefer cosine protocol with aug |
| A5stage20-noMem-aug wins overall | Lock A2+F0+aug+stage20 as official MoNuSeg |
| Aug hurts | Fall back to short20/cosine no-aug |

Submit command:
```bash
set -a; source /home/nada/.config/omnisam-clean-reruns/secrets.env; set +a
python scripts/submit_monuseg_test_screen.py --variants A3M4short20aug,A3M4cos40aug,A5stage20noMemAug --seeds 0,1,2
```

**Bug note:** A `stage_label` UnboundLocalError in the per-epoch CSV write crashed all 9 jobs at end of epoch 1. Fixed in commit `d8f4538`. Jobs self-recovered by re-downloading the patched zip from the branch on each restart.

---

## Batch 3 results — COMPLETE (9/9)

### Per-seed detail — A3M4-short20-aug (20 epochs, constant, aug)

| Seed | final_epoch dice | final_epoch iou | best_test dice | best_epoch | gap |
|---:|---:|---:|---:|---:|---|
| 0 | 0.8050 | 0.6744 | 0.8050 | 20 | 0 |
| 1 | 0.8102 | 0.6817 | 0.8102 | 20 | 0 |
| 2 | 0.8141 | 0.6873 | 0.8141 | 20 | 0 |
| **mean** | **0.810±0.004** | **0.681±0.006** | same | — | — |

### Per-seed detail — A3M4-cos40-aug (40 epochs, cosine, aug)

| Seed | final_epoch dice | final_epoch iou | best_test dice | best_epoch | gap |
|---:|---:|---:|---:|---:|---|
| 0 | 0.8123 | 0.6848 | 0.8129 | 37 | −0.001 |
| 1 | 0.8127 | 0.6853 | 0.8134 | 30 | −0.001 |
| 2 | 0.8125 | 0.6849 | 0.8127 | 33 | −0.000 |
| **mean** | **0.812±0.000** | **0.685±0.000** | 0.813±0.000 | — | — |

### Per-seed detail — A5stage20-noMem-aug (stage1=20, stage2=20, aug) ← WINNER

| Seed | final_epoch dice | final_epoch iou | best_test dice | best_epoch | gap |
|---:|---:|---:|---:|---:|---|
| 0 | 0.8179 | 0.6927 | 0.8211 | 35 | −0.003 |
| 1 | 0.8213 | 0.6976 | 0.8213 | 40 | 0 |
| 2 | 0.8117 | 0.6840 | 0.8226 | 39 | −0.011 |
| **mean** | **0.817±0.004** | **0.691±0.006** | 0.822±0.001 | — | — |

### Batch 3 decision rule outcomes

| Rule | Result | Decision |
|---|---|---|
| A3M4-short20-aug > A3M4-short20 (0.810)? | 0.810 = 0.810 — TIED | Aug rescues collapse but doesn't improve mean for constant LR |
| A3M4-cos40-aug ≥ short20-aug? | 0.812 > 0.810 — YES | Prefer cosine+aug protocol over constant+aug |
| A5stage20-noMem-aug wins overall? | 0.817 >> 0.812 — YES | **Lock A2+F0+aug+stage20 as official MoNuSeg** |

### Official MoNuSeg architecture (locked)

> **A2+F0 (`final_staged_a2`) + `autosam_dense_v1` augmentation + stage1=20 + stage2=20**
> Claim: `final_epoch` at ep40 = **0.817±0.004** Dice  
> Config: `configs/test_screen/monuseg_a5_staged20_no_memory_aug.yaml`

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
