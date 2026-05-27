# v6 Pending Jobs — Kill Audit

**Date:** 2026-05-27  
**Action:** Cancel 3 stuck Pending v6 jobs before clean resubmission.

---

## Jobs Identified

| Job name | Status | Age | Duration | GPU | Pods |
|----------|--------|-----|----------|-----|------|
| monuseg-ts80-v6-onestage-s0 | Pending | 4h | 0s | 0/1 | 0 running, 1 pending |
| monuseg-ts80-v6-onestage-s1 | Pending | 4h | 0s | 0/1 | 0 running, 1 pending |
| monuseg-ts80-v6-onestage-s2 | Pending | 4h | 0s | 0/1 | 0 running, 1 pending |

## Why stuck / Pending

- `Duration: 0s` and `Events: No events for resources` for all three jobs.
- Pods show `STATUS: PENDING` with `NODE: N/A` — never scheduled.
- 4 hours in queue with no allocation suggests cluster capacity exhausted for the project or scheduling delay beyond the `--pod-running-timeout 20m` hint (RunAI honors this as a preference, not a hard kill for Pending).
- v1–v5 all ran successfully (3–4 hours each), so this is likely queue depth: by the time v6 was queued, GPU slots were full. RunAI does not auto-cancel Pending jobs on timeout.

## Output directory status

- Expected path: `/storage/nada/test_screen_80/v6_onestage_gated/s{0,1,2}/`
- PVC not mounted locally; no artifacts can be confirmed.
- Since `Duration: 0s` and no pod ever ran, no output artifacts exist.

## W&B status

- W&B group `monuseg_testscreen80_layerwarm_aug` — no v6 runs visible (jobs never started, so W&B init was never called).

## Submitted command (from `runai describe`)

```
runai submit --name monuseg-ts80-v6-onestage-s0 \
  --image pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime \
  --gpu 1 --project avidan --cpu 4 --memory 32G \
  --existing-pvc claimname=storage,path=/storage \
  --working-dir /storage/nada \
  --git-sync source=https://github.com/nadavo11/omnisam-clean-launcher.git,branch=main,target=/storage/nada/ts80l/v6g/s0 \
  ...
  --command -- python .../launch_monuseg_80_layerwarm.py \
    --run-name monuseg_ts80_v6_onestage_s0 \
    --human-name full_A3M4_F0_from_epoch0_gated_aug80 \
    --config configs/test_screen_80_layerwarm/v6_full_a3m4_f0_from_epoch0_gated_aug80.yaml \
    --seed 0 \
    --output-dir /storage/nada/test_screen_80/v6_onestage_gated/s0 \
    --wandb-group monuseg_testscreen80_layerwarm_aug \
    --wandb-job-type TEST_SCREEN_80
```

## Action taken

All three v6 jobs cancelled via `runai delete job --name <job> -p avidan`.

## Resubmission plan

- New job names: `monuseg-ts80-v6r-s{0,1,2}` (suffix `-v6r` distinguishes resub)
- New output paths: `/storage/nada/verify/v6_resub/s{seed}/`
- Same config: `configs/test_screen_80_layerwarm_verify/v6_resub_aug80.yaml`
- Run names: `monuseg_testscreen80_v6_from_epoch0_gated_resub_s{seed}`
- W&B group: `monuseg_testscreen80_layerwarm_aug`
