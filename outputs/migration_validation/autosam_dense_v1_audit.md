# autosam_dense_v1 Augmentation Port Audit

**Date**: 2026-05-25  
**Source**: `src/rwtd_sam3/eval/frozen_mask_head_augmentations.py`  
**Migrated to**: `src/frozen_sam_readout/training/augmentations.py`

---

## Augmentation policy summary

Policy name: `autosam_dense_v1`

Applied: **training time only** — eval uses clean resized image.  
Applied: **before SAM feature extraction** — the augmented PIL image is passed to the SAM encoder.  
Applied: **jointly to image and binary mask** — spatial transforms use the same angle/scale/flip.

### Operations (in fixed order, then randomised sub-order for color jitter)

| Step | Operation | Parameters |
|---|---|---|
| 1 | Affine (rotation + scale) | angle ~ Uniform[-20°, 20°], scale ~ Uniform[0.75, 1.25], translate=0, shear=0, fill=0 |
| 2 | Horizontal flip | p=0.5 |
| 3 | Color jitter (random order) | brightness ∈ [0.6, 1.4], contrast ∈ [0.6, 1.4], saturation ∈ [0.6, 1.4], hue ∈ [-0.1, 0.1] |

Color jitter: each factor is sampled uniformly then the 4 ops (brightness, contrast, saturation, hue) are applied in a **uniformly random permutation** — same as torchvision ColorJitter default behavior.

### Exact parameter values (from source `AUTOSAM_DENSE_AUGMENTATION_SETTINGS`)

```python
AUTOSAM_DENSE_V1_SETTINGS = {
    "rotation_degrees": 20.0,
    "scale_min": 0.75,
    "scale_max": 1.25,
    "horizontal_flip_probability": 0.5,
    "brightness": 0.4,
    "contrast": 0.4,
    "saturation": 0.4,
    "hue": 0.1,
}
```

### Affine implementation (exact port)

Uses PIL `Image.transform(AFFINE, ...)` with centre-aligned rotation+scale matrix:

```python
cx, cy = width / 2.0, height / 2.0
rad = np.deg2rad(angle)
cos_t = cos(rad) * scale
sin_t = sin(rad) * scale
a, b, c = cos_t, sin_t, (1 - cos_t)*cx - sin_t*cy
d, e, f = -sin_t, cos_t, sin_t*cx + (1 - cos_t)*cy
image.transform((W, H), AFFINE, (a, b, c, d, e, f), resample=BILINEAR, fillcolor=0)
mask.transform((W, H), AFFINE, (a, b, c, d, e, f), resample=NEAREST, fillcolor=0)
```

### Hue adjustment (exact port)

Uses `colorsys.rgb_to_hsv` / `colorsys.hsv_to_rgb` per-pixel — intentionally slow but numerically exact match to source.

### Validity check

After each augmentation attempt, the result is discarded if the augmented mask has no foreground or no background pixels. Up to `max_attempts=4` retries are made before returning the original sample.

---

## When augmentation is active

| Dataset | Augmentation | Source |
|---|---|---|
| GlaS training (official A0, A2) | `autosam_dense_v1` | confirmed in source job submissions |
| MoNuSeg training | `none` | MoNuSeg used no augmentation in official runs |
| Eval (any dataset) | `none` | augmentation disabled at eval |

GlaS official anchor A0=0.9374 was trained with `autosam_dense_v1`. Any reproduction
of GlaS results must activate this policy for training.

---

## Impact on feature caching

When `autosam_dense_v1` is active, **features cannot be pre-cached** because each
epoch uses a freshly-augmented version of each image. The training loop must run
SAM feature extraction on-the-fly for each batch. This matches the source-repo
behavior (`skip_feature_materialization=(augmentation_policy != "none")`).

In the migrated repo, `train_stage1.py` and `train_stage2.py` check the
augmentation config and skip the feature cache accordingly.

---

## Verification test

`tests/test_augmentation.py` (to be added) must:
1. Load a synthetic RGB image and binary mask.
2. Run `apply_augmentation` with `autosam_dense_v1` and a fixed seed.
3. Assert output is a different PIL image (not identical to input).
4. Assert augmented mask covers > 0 pixels in both fg and bg.
5. Assert image dimensions are preserved.
6. Assert that with `policy="none"`, the sample is returned unchanged.
