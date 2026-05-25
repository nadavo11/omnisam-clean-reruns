"""A5: all-at-once joint training of fpn_2 + fpn_1 + fpn_0 with learned alpha.

Same architecture as the final staged head, but trained jointly from scratch
(no two-stage warm-start). Ablation only.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..blocks.learned_alpha import LearnedAlphaResidual
from .a3_memory import A3MemoryHead


class A5F0ResidualHead(nn.Module):
    def __init__(
        self,
        f2_channels: int,
        f1_channels: int,
        f0_channels: int,
        decoder_dim: int = 128,
        memory_tokens: int = 8,
        alpha_init: float = 0.05,
    ) -> None:
        super().__init__()
        self.a3 = A3MemoryHead(
            f2_channels=f2_channels,
            f1_channels=f1_channels,
            decoder_dim=decoder_dim,
            memory_tokens=memory_tokens,
        )
        self.f0_residual = LearnedAlphaResidual(
            f0_channels=f0_channels,
            decoder_dim=decoder_dim,
            alpha_init=alpha_init,
            final_layer_zero_init=True,
        )

    def forward(self, pyramid: dict[str, torch.Tensor]) -> torch.Tensor:
        l_a3 = self.a3(pyramid)
        return self.f0_residual(pyramid["fpn_0"], l_a3)
