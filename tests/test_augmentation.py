"""Tests for autosam_dense_v1 augmentation port."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from frozen_sam_readout.training.augmentations import (
    AUTOSAM_DENSE_V1_SETTINGS,
    apply_augmentation,
    describe_augmentation_policy,
)


def _make_sample(h: int = 64, w: int = 64):
    """Build a minimal sample object with image and mask fields."""
    from types import SimpleNamespace

    rng = np.random.default_rng(0)
    img_arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    fg = np.zeros((h, w), dtype=bool)
    fg[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = True
    bg = ~fg
    return SimpleNamespace(
        image=Image.fromarray(img_arr, mode="RGB"),
        texture_a_mask=fg,
        texture_b_mask=bg,
    )


class TestDescribePolicy:
    def test_none(self):
        assert describe_augmentation_policy("none") == "none"

    def test_autosam_dense_v1_contains_key_params(self):
        desc = describe_augmentation_policy("autosam_dense_v1")
        assert "affine" in desc
        assert "hflip" in desc
        assert "color_jitter" in desc

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            describe_augmentation_policy("bad_policy")


class TestApplyAugmentation:
    def test_none_returns_same_object(self):
        sample = _make_sample()
        rng = np.random.default_rng(42)
        result = apply_augmentation(sample, rng=rng, policy="none")
        assert result is sample

    def test_autosam_dense_v1_changes_image(self):
        sample = _make_sample()
        rng = np.random.default_rng(7)
        result = apply_augmentation(sample, rng=rng, policy="autosam_dense_v1")
        orig_arr = np.asarray(sample.image)
        aug_arr = np.asarray(result.image)
        assert orig_arr.shape == aug_arr.shape, "Dimensions must be preserved"
        assert not np.array_equal(orig_arr, aug_arr), "Augmented image should differ from original"

    def test_autosam_dense_v1_mask_valid(self):
        sample = _make_sample()
        rng = np.random.default_rng(13)
        result = apply_augmentation(sample, rng=rng, policy="autosam_dense_v1")
        fg = np.asarray(result.texture_a_mask, dtype=bool)
        bg = np.asarray(result.texture_b_mask, dtype=bool)
        assert fg.any(), "Augmented foreground must have ≥1 pixel"
        assert bg.any(), "Augmented background must have ≥1 pixel"
        assert not np.logical_and(fg, bg).any(), "fg and bg must be disjoint"

    def test_settings_unchanged(self):
        assert AUTOSAM_DENSE_V1_SETTINGS["rotation_degrees"] == 20.0
        assert AUTOSAM_DENSE_V1_SETTINGS["scale_min"] == 0.75
        assert AUTOSAM_DENSE_V1_SETTINGS["scale_max"] == 1.25
        assert AUTOSAM_DENSE_V1_SETTINGS["horizontal_flip_probability"] == 0.5
        assert AUTOSAM_DENSE_V1_SETTINGS["brightness"] == 0.4
        assert AUTOSAM_DENSE_V1_SETTINGS["hue"] == 0.1
