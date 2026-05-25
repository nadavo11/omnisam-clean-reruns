"""Shape-only smoke test for head variants under a synthetic pyramid."""

import torch

from frozen_sam_readout.models.registry import build_head


def _fake_pyramid():
    return {
        "fpn_2": torch.randn(1, 256, 8, 8),
        "fpn_1": torch.randn(1, 64, 16, 16),
        "fpn_0": torch.randn(1, 32, 32, 32),
    }


def test_a0_shape():
    head = build_head("a0_fpn2", f2_channels=256)
    out = head(_fake_pyramid())
    assert out.shape == (1, 1, 8, 8)


def test_a2_shape():
    head = build_head("a2_fpn2_fpn1_refine", f2_channels=256, f1_channels=64)
    out = head(_fake_pyramid())
    assert out.shape == (1, 1, 16, 16)


def test_a3_shape():
    head = build_head("a3_memory", f2_channels=256, f1_channels=64, memory_tokens=4)
    out = head(_fake_pyramid())
    assert out.shape == (1, 1, 16, 16)


def test_final_staged_shape_and_alpha():
    head = build_head(
        "final_staged",
        f2_channels=256, f1_channels=64, f0_channels=32, memory_tokens=4,
    )
    out = head(_fake_pyramid())
    assert out.shape == (1, 1, 32, 32)
    assert 0.0 < float(head.alpha) < 1.0
    # alpha_init=0.05 -> sigmoid(raw_alpha)≈0.05
    assert abs(float(head.alpha) - 0.05) < 1e-3
