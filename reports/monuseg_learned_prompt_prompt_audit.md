# MoNuSeg Learned Prompt Type Audit

Insertion-point classification used in this batch:
- `FPNplusD0_box_plus_learned_delta_aug80`: `sparse`
- `FPNplusD0_learned_sparse_prompt_aug80`: `sparse`
- `FPNplusD0_box_plus_learned_text_soft_prompt_aug80_s0`: `text`
- `post-decoder`: not used

Prompt-type audit table:

| variant | prompt_category | tensor_name | tensor_shape | trainable_shape | prompt_param_count | spatial_or_token | shared_across_images | image_conditioned | gt_derived | initialized_from | notes |
| --- | --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| `FPNplusD0_box_plus_learned_delta_aug80` | `sparse` | `geometry_encoder.geo_feats` | `[2, 1, 256]` | `[1, 1, 256]` | 256 | `token` | yes | no | no | `fixed_full_image_box_geo_prompt + zero residual delta` | delta is broadcast across the 2 box-derived geometry tokens |
| `FPNplusD0_learned_sparse_prompt_aug80` | `sparse` | `visual_prompt_embed` | `[4, 1, 256]` | `[4, 1, 256]` | 1024 | `token` | yes | no | no | `normal(0,0.02)` | constant learned task tokens concatenated into prompt sequence |
| `FPNplusD0_box_plus_learned_text_soft_prompt_aug80_s0` | `text` | `language_features_soft_prompt_tokens` | `[4, 1, 256]` | `[4, 1, 256]` | 1024 | `token` | yes | no | no | `normal(0,0.02)` | exploratory seed-0 text-side soft prompt concatenated after fixed text tokens |

Required zero-delta equivalence check:
- variant: `FPNplusD0_box_plus_learned_delta_aug80`
- category: `sparse`
- result:
  - `max_abs_diff = 0.0`
  - `mean_abs_diff = 0.0`
  - `cosine_similarity = 1.000601053237915`

Interpretation:
- The preferred `box_plus_learned_delta` implementation is a true sparse/token prompt ablation.
- It modifies prompt-token embeddings before the SAM3 grounding encoder/decoder stack.
- It does not modify `D0` directly, so it is not a post-decoder feature hack.
