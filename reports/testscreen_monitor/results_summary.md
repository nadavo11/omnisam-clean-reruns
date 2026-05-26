# MoNuSeg TEST_SCREEN Ablation — Final Results

**Date:** 2026-05-26  
**Protocol:** TEST_SCREEN — train on full MoNuSeg training set (37 patches), evaluate on official held-out test split (14 patches) every epoch, checkpoint selected by best test Dice. Acknowledged test-screening; not unbiased final evaluation.  
**W&B group:** `monuseg_test_screen_ablation` (W&B was offline for all runs due to project-name bug; metrics from training logs only)  
**Evaluator:** `run_official_eval()` — direct foreground Dice/IoU, threshold=0.5, upsampled to GT resolution  

---

## Per-Run Results

| Run | Variant | Seed | best_dice | best_iou | best_epoch | Notes |
|---|---|---:|---:|---:|---:|---|
| monuseg-testscreen-a3m4-s0 | A3M4 | 0 | 0.8086 | 0.6794 | 20 | ✅ Succeeded |
| monuseg-testscreen-a3m4-s1 | A3M4 | 1 | 0.8095 | 0.6807 | 19 | ✅ Succeeded |
| monuseg-testscreen-a3m4-s2 | A3M4 | 2 | 0.8111 | 0.6830 | 21 | ✅ Succeeded |
| monuseg-testscreen-a3m8-s0 | A3M8 | 0 | 0.8066 | 0.6766 | 17 | ✅ Succeeded |
| monuseg-testscreen-a3m8-s1 | A3M8 | 1 | 0.8095 | 0.6807 | 19 | ✅ Succeeded |
| monuseg-testscreen-a3m8-s2 | A3M8 | 2 | 0.8088 | 0.6798 | 18 | ✅ Succeeded |
| monuseg-testscreen-a5staged-from-a3-s0 | A5staged | 0 | 0.8177 | 0.6924 | — | ✅ Succeeded (RunAI shows Error = pod cleanup) |
| monuseg-testscreen-a5staged-from-a3-s1 | A5staged | 1 | 0.8181 | 0.6930 | — | ✅ Succeeded |
| monuseg-testscreen-a5staged-from-a3-s2 | A5staged | 2 | 0.8121 | 0.6845 | — | ✅ Succeeded |
| monuseg-testscreen-a5staged-no-memory-s0 | A5noMem | 0 | 0.8160 | 0.6901 | — | ✅ Succeeded (RunAI shows Error = pod cleanup) |
| monuseg-testscreen-a5staged-no-memory-s1 | A5noMem | 1 | 0.8131 | 0.6861 | — | ✅ Succeeded (RunAI shows Error = pod cleanup) |
| monuseg-testscreen-a5staged-no-memory-s2 | A5noMem | 2 | 0.8155 | 0.6893 | — | ✅ Succeeded |

*best_epoch for A5staged/A5noMem not extracted — two-stage runs require cross-referencing per-epoch logs.*

---

## Aggregate by Variant

| Variant | Description | Seeds | Dice mean ± std | IoU mean ± std | Peak best_dice |
|---|---|---:|---:|---:|---:|
| **A3M4** | a3_memory, memory_tokens=4, 40 epochs | 0,1,2 | **0.8097 ± 0.0013** | **0.6810 ± 0.0018** | 0.8111 |
| **A3M8** | a3_memory, memory_tokens=8, 40 epochs | 0,1,2 | **0.8083 ± 0.0015** | **0.6790 ± 0.0021** | 0.8095 |
| **A5staged** | final_staged (A3+F0), 2-stage 40+40 epochs | 0,1,2 | **0.8160 ± 0.0031** | **0.6900 ± 0.0044** | 0.8181 |
| **A5noMem** | final_staged_a2 (A2+F0), 2-stage 40+40 epochs | 0,1,2 | **0.8149 ± 0.0015** | **0.6885 ± 0.0021** | 0.8160 |

---

## Three-view checkpoint results

### `final_epoch` — claimable within TEST_SCREEN protocol

| Variant | s0 dice | s1 dice | s2 dice | mean ± std | s0 iou | s1 iou | s2 iou | mean ± std |
|---|---|---|---|---|---|---|---|---|
| A5noMem | 0.8116 | 0.8075 | 0.8059 | **0.808 ± 0.003** | 0.6838 | 0.6782 | 0.6761 | **0.680 ± 0.004** |
| A5staged | 0.8051 | 0.8057 | 0.7982 | **0.803 ± 0.004** | 0.6747 | 0.6754 | 0.6652 | **0.672 ± 0.005** |
| A3M8 | 0.7903 | 0.8011 | 0.7889 | 0.793 ± 0.006 | 0.6542 | 0.6691 | 0.6523 | 0.659 ± 0.007 |
| A3M4 | 0.7959 | 0.7884 | 0.7951 | 0.793 ± 0.003 | 0.6619 | 0.6518 | 0.6608 | 0.658 ± 0.006 |

### `best_test_epoch` — diagnostic oracle only (not claimable)

| Variant | s0 dice (ep) | s1 dice (ep) | s2 dice (ep) | mean ± std |
|---|---|---|---|---|
| A5staged | 0.8177 (63) | 0.8181 (72) | 0.8121 (79) | **0.816 ± 0.003** |
| A5noMem | 0.8160 (53) | 0.8131 (43) | 0.8155 (53) | **0.815 ± 0.002** |
| A3M4 | 0.8086 (20) | 0.8095 (19) | 0.8111 (21) | 0.810 ± 0.001 |
| A3M8 | 0.8066 (17) | 0.8095 (19) | 0.8088 (18) | 0.808 ± 0.002 |

### `fixed_epoch_20` for A3 variants (post-hoc, not predeclared)

| Variant | s0 | s1 | s2 | mean ± std |
|---|---|---|---|---|
| A3M4 | 0.8086 | 0.8094 | 0.8110 | **0.810 ± 0.001** |
| A3M8 | 0.8045 | 0.8094 | 0.8083 | **0.807 ± 0.003** |

---

## Key Findings

1. **Best claimable result: A5noMem final_epoch 0.808 ± 0.003** — A2+F0, stage1=40, stage2=40, constant LR.
2. **Best diagnostic oracle: A5staged best_test_epoch 0.816 ± 0.003** — not claimable; selected by test performance.
3. **F0 residual is the main gain:** A5noMem final_epoch (+0.015 Dice over A3M4 final_epoch).
4. **Memory tokens add nothing in 2-stage setting:** A5noMem ≥ A5staged. Simpler architecture preferred.
5. **Stage1=40 is too long:** A3 peaks at ep17–21; A5noMem-s1 achieves best at stage2-ep3, confirming rapid recovery after optimizer re-init.
6. **Constant LR gap:** final_epoch@40 is 0.017 Dice below best_test_epoch for A3 variants. Cosine schedule should close this.
7. **W&B offline** for all 12 runs (project-name bug fixed). Batch 2 will have W&B online.

---

## Caveats

- TEST_SCREEN: all numbers involve test-split visibility during training — not unbiased final evaluation.
- 14 test samples: ±0.007 Dice per 1-sample flip — small absolute differences may be noise.
- `final_epoch` is the claim-bearing metric; `best_test_epoch` is an oracle diagnostic ceiling.
- W&B curves and segmentation visuals unavailable for batch 1 (offline mode + ephemeral pod storage).
- Backend: HF transformers fallback (all FPN levels 256ch) — not paper-headline-safe.

---

## Next batch (batch 2)

Configs ready, submit command:
```bash
python scripts/submit_monuseg_test_screen.py --seeds 0,1,2
# submits: A3M4short20, A3M4cos40, A5stage20fromA3M4, A5stage20noMem × seeds 0/1/2
```

Full reports: `reports/e2_monuseg_test_ablation_screen.md`
