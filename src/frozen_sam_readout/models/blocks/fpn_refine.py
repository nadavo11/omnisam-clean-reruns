"""F1 residual refine block fed by F2 logits and F1 features."""

from __future__ import annotations

import torch
import torch.nn as nn

from .conv_blocks import ConvBNReLU
from .upsample import upsample_to


class FpnRefineBlock(nn.Module):
    """Take F1 features + coarse F2 logits and produce a refined logit map."""

    def __init__(
        self,
        f1_channels: int,
        decoder_dim: int = 128,
        zero_init_final: bool = False,
    ) -> None:
        super().__init__()
        self.f1_proj = nn.Conv2d(f1_channels, decoder_dim, kernel_size=1)
        self.fuse = nn.Sequential(
            ConvBNReLU(decoder_dim + 1, decoder_dim),
            ConvBNReLU(decoder_dim, decoder_dim),
        )
        self.head = nn.Conv2d(decoder_dim, 1, kernel_size=1)
        if zero_init_final:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, f1: torch.Tensor, coarse_logits: torch.Tensor) -> torch.Tensor:
        h, w = f1.shape[-2:]
        coarse_up = upsample_to(coarse_logits, (h, w))
        z = torch.cat([self.f1_proj(f1), coarse_up], dim=1)
        return self.head(self.fuse(z))
