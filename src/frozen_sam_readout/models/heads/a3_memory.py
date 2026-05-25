"""A3: A2 + global memory cross-attention applied to the fpn_2 features."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..blocks.conv_blocks import ConvBNReLU
from ..blocks.fpn_refine import FpnRefineBlock
from ..blocks.memory_attention import MemoryAttentionBlock


class A3MemoryHead(nn.Module):
    def __init__(
        self,
        f2_channels: int,
        f1_channels: int,
        decoder_dim: int = 128,
        memory_tokens: int = 8,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(f2_channels, decoder_dim, kernel_size=1)
        self.memory = MemoryAttentionBlock(
            embed_dim=decoder_dim,
            num_memory_tokens=memory_tokens,
            num_heads=num_heads,
        )
        self.body = nn.Sequential(
            ConvBNReLU(decoder_dim, decoder_dim),
            ConvBNReLU(decoder_dim, decoder_dim),
        )
        self.coarse_head = nn.Conv2d(decoder_dim, 1, kernel_size=1)
        self.refine = FpnRefineBlock(f1_channels, decoder_dim=decoder_dim)

    def coarse_logits(self, pyramid: dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.proj(pyramid["fpn_2"])
        x = self.memory(x)
        return self.coarse_head(self.body(x))

    def forward(self, pyramid: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.refine(pyramid["fpn_1"], self.coarse_logits(pyramid))
