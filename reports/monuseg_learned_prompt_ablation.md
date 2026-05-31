# MoNuSeg Learned Constant Prompt Ablation

Date: 2026-05-31

Status:
- Code path implemented for `learned_constant_sparse` and `fixed_full_image_box_plus_learned_delta`.
- Local one-epoch smoke passed for:
  - `FPNplusD0_box_plus_learned_delta_aug80` seed 0
  - `FPNplusD0_learned_sparse_prompt_aug80` seed 0
- RunAI batch submission is currently blocked on this host:
  - `runai submit ...`
  - error: `dial tcp 127.0.0.1:8080: connect: connection refused`

Reference fixed-box baseline:
- Variant: `FPNplusD0_aug80`
- Final Dice: `0.8291 ± 0.0003`
- Final IoU: `0.7090 ± 0.0004`
- Best seed: `0`
- W&B: `https://wandb.ai/nadavoteam/frozen-sam-readout/runs/56u3cpjl`

Implemented variants:
- `FPNplusD0_boxprompt_aug80_reference`
- `FPNplusD0_learned_sparse_prompt_aug80`
- `FPNplusD0_box_plus_learned_delta_aug80`
- `D0small_box_plus_learned_delta_aug80`

Smoke evidence:
- Delta smoke output: `/home/nada/PycharmProjects/omniSAM/outputs/smoke_learned_prompt_delta_s0`
- Sparse smoke output: `/home/nada/PycharmProjects/omniSAM/outputs/smoke_learned_prompt_sparse_s0`
- Delta smoke W&B: `https://wandb.ai/nadavoteam/frozen-sam-readout/runs/faxm13zk`
- Sparse smoke W&B: `https://wandb.ai/nadavoteam/frozen-sam-readout/runs/1is23kf5`

Key smoke checks:
- Zero-delta equivalence against legacy fixed-box D0:
  - `max_abs_diff = 0.0`
  - `mean_abs_diff = 0.0`
  - `cosine_similarity = 1.000601053237915`
- Trainability:
  - frozen SAM trainable params: `0`
  - trainable head params: `656,513`
  - trainable prompt params: `256` for delta mode
  - optimizer params: head + prompt only
- Prompt constraints:
  - shared across images: `true`
  - image-conditioned: `false`
  - GT-derived: `false`
  - prediction-derived at test: `false`

RunAI batch intended:
- Variants:
  - `FPNplusD0_box_plus_learned_delta_aug80`
  - `FPNplusD0_learned_sparse_prompt_aug80`
  - `D0small_box_plus_learned_delta_aug80`
- Seeds: `0,1,2`
- Submission script:
  - `/home/nada/PycharmProjects/omniSAM/scripts/submit_monuseg_learned_prompt_runai.py`
- Branch pushed for cluster sync:
  - `codex/clean-e1-runai-wandb`
  - commit `58284ce`
