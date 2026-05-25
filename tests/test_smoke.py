"""Top-level import / construction smoke checks."""

import torch


def test_imports():
    import frozen_sam_readout  # noqa: F401
    from frozen_sam_readout import data, evaluation, models, sam, training, utils, visualization  # noqa: F401
    from frozen_sam_readout.models.registry import MODEL_REGISTRY

    expected = {"a0_fpn2", "a2_fpn2_fpn1_refine", "a3_memory", "a5_all_at_once", "final_staged"}
    assert expected.issubset(MODEL_REGISTRY.keys())


def test_loss_and_forward_smoke():
    from frozen_sam_readout.models.registry import build_head
    from frozen_sam_readout.training import bce_dice_loss

    head = build_head(
        "final_staged",
        f2_channels=256, f1_channels=64, f0_channels=32, memory_tokens=4,
    )
    pyramid = {
        "fpn_2": torch.randn(1, 256, 4, 4),
        "fpn_1": torch.randn(1, 64, 8, 8),
        "fpn_0": torch.randn(1, 32, 16, 16),
    }
    logits = head(pyramid)
    target = torch.zeros_like(logits)
    target[..., :8, :8] = 1.0
    loss = bce_dice_loss(logits, target)
    assert torch.isfinite(loss)
    loss.backward()
