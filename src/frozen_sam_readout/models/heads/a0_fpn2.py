"""A0: coarse-only readout from fpn_2."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..blocks.conv_blocks import ConvBNReLU


class A0Fpn2Head(nn.Module):
    """1x1 projection + 2x ConvBNReLU + 1x1 logits, on fpn_2 only."""

    def __init__(self, f2_channels: int, decoder_dim: int = 128) -> None:
        super().__init__()
        self.proj = nn.Conv2d(f2_channels, decoder_dim, kernel_size=1)
        self.body = nn.Sequential(
            ConvBNReLU(decoder_dim, decoder_dim),
            ConvBNReLU(decoder_dim, decoder_dim),
        )
        self.head = nn.Conv2d(decoder_dim, 1, kernel_size=1)

    def forward(self, pyramid: dict[str, torch.Tensor]) -> torch.Tensor:
        f2 = pyramid["fpn_2"]
        return self.head(self.body(self.proj(f2)))
