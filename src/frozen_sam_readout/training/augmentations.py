"""Train-time augmentations for the frozen-SAM readout.

The ``autosam_dense_v1`` policy is the official augmentation used for GlaS
training and is an exact port of the source-repo implementation in
``rwtd_sam3.eval.frozen_mask_head_augmentations``.

Policy summary:
  affine(angle_uniform[-20,20], scale_uniform[0.75,1.25], translate=0, shear=0, fill=0)
  + hflip_p0.5
  + color_jitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
  + operations applied in random order
  + applied to PIL image and binary mask BEFORE SAM feature extraction
  + training only (eval uses the unaugmented image)

Ports of source:
  src/rwtd_sam3/eval/frozen_mask_head_augmentations.py
"""

from __future__ import annotations

import colorsys
from dataclasses import replace
from typing import Any

import numpy as np
from PIL import Image

SUPPORTED_POLICIES = ("none", "autosam_dense_v1")

# Exact parameter values from source repo — do not change without updating audit.
AUTOSAM_DENSE_V1_SETTINGS: dict[str, float] = {
    "rotation_degrees": 20.0,
    "scale_min": 0.75,
    "scale_max": 1.25,
    "horizontal_flip_probability": 0.5,
    "brightness": 0.4,
    "contrast": 0.4,
    "saturation": 0.4,
    "hue": 0.1,
}


def describe_augmentation_policy(policy: str) -> str:
    if policy == "none":
        return "none"
    if policy == "autosam_dense_v1":
        return (
            "autosam_dense_v1:"
            "affine(angle_uniform[-20,20],scale_uniform[0.75,1.25],translate=0,shear=0,fill=0)"
            "+hflip_p0.5"
            "+color_jitter(brightness=0.4,contrast=0.4,saturation=0.4,hue=0.1)"
            "+train_only"
        )
    raise ValueError(f"Unknown augmentation policy '{policy}'. Supported: {SUPPORTED_POLICIES}")


def apply_augmentation(
    sample: Any,
    *,
    rng: np.random.Generator,
    policy: str,
    max_attempts: int = 4,
) -> Any:
    """Apply an augmentation policy to a binary segmentation sample.

    The sample must have ``.image`` (PIL Image), ``.texture_a_mask`` (foreground
    bool array), and ``.texture_b_mask`` (background bool array) attributes, matching
    the interface of ``MonusegBinarySample`` / ``GlasBinarySample`` from the source repo.

    Augmentation operates on the PIL image and binary mask BEFORE SAM feature
    extraction. It is applied only at training time.

    Args:
        sample: A binary segmentation sample with image and mask fields.
        rng: NumPy random generator for reproducibility.
        policy: One of ``SUPPORTED_POLICIES``.
        max_attempts: Retry count if augmented mask becomes all-foreground or all-background.

    Returns:
        Augmented sample (same type, replaced fields). Returns original if max_attempts
        exceeded without a valid augmented mask.
    """

    if policy == "none":
        return sample
    if policy != "autosam_dense_v1":
        raise ValueError(f"Unknown augmentation policy '{policy}'.")
    for _ in range(max(1, int(max_attempts))):
        augmented = _apply_autosam_dense_v1(sample=sample, rng=rng)
        fg = np.asarray(augmented.texture_a_mask, dtype=bool)
        bg = np.asarray(augmented.texture_b_mask, dtype=bool)
        if fg.any() and bg.any():
            return augmented
    return sample


def _apply_autosam_dense_v1(sample: Any, *, rng: np.random.Generator) -> Any:
    s = AUTOSAM_DENSE_V1_SETTINGS
    angle = float(rng.uniform(-s["rotation_degrees"], s["rotation_degrees"]))
    scale = float(rng.uniform(s["scale_min"], s["scale_max"]))
    do_hflip = bool(rng.random() < s["horizontal_flip_probability"])

    # Convert foreground mask to uint8 PIL image for joint spatial transform.
    fg_mask_arr = np.asarray(sample.texture_a_mask, dtype=np.uint8) * 255
    fg_mask_img = Image.fromarray(fg_mask_arr, mode="L")

    # Apply affine (rotation + scale, no translate, no shear, fill=0).
    image = _affine_pil(sample.image, angle=angle, scale=scale, resample=Image.Resampling.BILINEAR)
    mask_img = _affine_pil(fg_mask_img, angle=angle, scale=scale, resample=Image.Resampling.NEAREST)

    if do_hflip:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mask_img = mask_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    # Color jitter with uniformly sampled factors in random order.
    brightness_f = float(rng.uniform(max(0.0, 1.0 - s["brightness"]), 1.0 + s["brightness"]))
    contrast_f = float(rng.uniform(max(0.0, 1.0 - s["contrast"]), 1.0 + s["contrast"]))
    saturation_f = float(rng.uniform(max(0.0, 1.0 - s["saturation"]), 1.0 + s["saturation"]))
    hue_f = float(rng.uniform(-s["hue"], s["hue"]))

    jitter_ops = [
        ("brightness", brightness_f),
        ("contrast", contrast_f),
        ("saturation", saturation_f),
        ("hue", hue_f),
    ]
    for op_idx in rng.permutation(len(jitter_ops)).tolist():
        op_name, op_val = jitter_ops[int(op_idx)]
        if op_name == "brightness":
            image = _adjust_brightness(image, op_val)
        elif op_name == "contrast":
            image = _adjust_contrast(image, op_val)
        elif op_name == "saturation":
            image = _adjust_saturation(image, op_val)
        elif op_name == "hue":
            image = _adjust_hue(image, op_val)

    aug_fg = np.asarray(mask_img, dtype=np.uint8) > 0
    aug_bg = np.logical_not(aug_fg)

    updates: dict = {
        "image": image,
        "texture_a_mask": aug_fg.astype(bool),
        "texture_b_mask": aug_bg.astype(bool),
    }

    if hasattr(sample, "boundary_mask"):
        from frozen_sam_readout.data.common import boundary_from_region_masks
        updates["boundary_mask"] = boundary_from_region_masks(aug_fg, aug_bg)

    # Try dataclasses.replace first (works for @dataclass objects).
    # Fall back to object __dict__ copy for SimpleNamespace / namedtuple-like objects.
    try:
        return replace(sample, **updates)
    except TypeError:
        import copy
        out = copy.copy(sample)
        for k, v in updates.items():
            setattr(out, k, v)
        return out


# --------------------------------------------------------------------------- #
#  Spatial transforms (exact port of source _apply_affine_pil)                #
# --------------------------------------------------------------------------- #

def _affine_pil(
    image: Image.Image,
    *,
    angle: float,
    scale: float,
    resample: int,
) -> Image.Image:
    """Centre-aligned affine: rotation + uniform scale, no translation, no shear."""
    width, height = image.size
    cx, cy = width / 2.0, height / 2.0
    rad = np.deg2rad(angle)
    cos_t = float(np.cos(rad) * scale)
    sin_t = float(np.sin(rad) * scale)
    a = cos_t
    b = sin_t
    c = (1.0 - a) * cx - b * cy
    d = -sin_t
    e = cos_t
    f = b * cx + (1.0 - e) * cy
    return image.transform(
        (width, height),
        Image.Transform.AFFINE,
        (a, b, c, d, e, f),
        resample=resample,
        fillcolor=0,
    )


# --------------------------------------------------------------------------- #
#  Colour jitter helpers (exact port of source _adjust_* functions)           #
# --------------------------------------------------------------------------- #

def _adjust_brightness(image: Image.Image, factor: float) -> Image.Image:
    from PIL import ImageEnhance
    return ImageEnhance.Brightness(image).enhance(float(factor))


def _adjust_contrast(image: Image.Image, factor: float) -> Image.Image:
    from PIL import ImageEnhance
    return ImageEnhance.Contrast(image).enhance(float(factor))


def _adjust_saturation(image: Image.Image, factor: float) -> Image.Image:
    from PIL import ImageEnhance
    return ImageEnhance.Color(image).enhance(float(factor))


def _adjust_hue(image: Image.Image, factor: float) -> Image.Image:
    """Per-pixel HSV hue shift via colorsys (exact port of source implementation)."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    rgb = np.asarray(image, dtype=np.uint8)
    flat = rgb.reshape(-1, 3).astype(np.float32) / 255.0
    hsv = np.array(
        [colorsys.rgb_to_hsv(float(p[0]), float(p[1]), float(p[2])) for p in flat],
        dtype=np.float32,
    )
    hsv[:, 0] = np.mod(hsv[:, 0] + float(factor), 1.0)
    rgb_out = np.array(
        [colorsys.hsv_to_rgb(float(p[0]), float(p[1]), float(p[2])) for p in hsv],
        dtype=np.float32,
    )
    rgb_out = np.clip(rgb_out.reshape(rgb.shape) * 255.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(rgb_out, mode="RGB")
