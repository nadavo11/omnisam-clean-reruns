"""Spatial upsampling helper."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def upsample_to(x: torch.Tensor, size_hw: tuple[int, int]) -> torch.Tensor:
    """Bilinearly upsample ``x`` to ``size_hw`` (H, W)."""

    return F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)
