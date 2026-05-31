# MoNuSeg Learned Prompt Audit

Current audit scope:
- local smoke only
- no multi-seed production runs yet
- no seed-0 post-training D0 visual audit yet

Smoke audit summary:
- Preferred delta mode is numerically identical to the legacy fixed-box D0 path at init:
  - `max_abs_diff = 0.0`
  - `mean_abs_diff = 0.0`
  - `cosine_similarity = 1.000601053237915`
- Sparse learned prompt initializes as a constant trainable token bank:
  - tokens: `4`
  - dim: `256`
  - params: `1024`
- Delta learned prompt initializes as a constant broadcast residual:
  - tokens: `1`
  - dim: `256`
  - params: `256`

Blocked next step:
- full seed-0 and multi-seed RunAI jobs could not be submitted from this host because the RunAI API endpoint at `127.0.0.1:8080` was unreachable.
