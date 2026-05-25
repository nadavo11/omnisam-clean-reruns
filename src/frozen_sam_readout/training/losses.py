"""BCE + Dice loss for binary foreground segmentation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def bce_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Symmetric BCE + soft-Dice loss; both inputs broadcast to logits' shape."""

    if target.shape != logits.shape:
        target = target.view_as(logits).to(logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, target.to(logits.dtype))
    probs = torch.sigmoid(logits)
    inter = (probs * target).sum()
    denom = probs.sum() + target.sum()
    dice = 1.0 - (2.0 * inter + eps) / (denom + eps)
    return bce_weight * bce + dice_weight * dice
