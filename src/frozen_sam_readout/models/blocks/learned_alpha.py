"""Learned-alpha residual block for the F0 stage of the staged readout.

R0 = F0Residual(F0, upsample(L_A3))
L_final = L_A3 + alpha * R0
alpha = sigmoid(raw_alpha), raw_alpha initialized so alpha ≈ 0.05.
The final 1x1 conv of the F0 residual head is zero-initialized so that at
the start of stage-2 training ``L_final ≈ L_A3``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .conv_blocks import ConvBNReLU
from .upsample import upsample_to


class LearnedAlphaResidual(nn.Module):
    def __init__(
        self,
        f0_channels: int,
        decoder_dim: int = 128,
        alpha_init: float = 0.05,
        final_layer_zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.f0_proj = nn.Conv2d(f0_channels, decoder_dim, kernel_size=1)
        self.fuse = nn.Sequential(
            ConvBNReLU(decoder_dim + 1, decoder_dim),
            ConvBNReLU(decoder_dim, decoder_dim),
        )
        self.final = nn.Conv2d(decoder_dim, 1, kernel_size=1)
        if final_layer_zero_init:
            nn.init.zeros_(self.final.weight)
            nn.init.zeros_(self.final.bias)
        alpha_init = float(min(max(alpha_init, 1e-4), 1.0 - 1e-4))
        raw = math.log(alpha_init / (1.0 - alpha_init))
        self.raw_alpha = nn.Parameter(torch.tensor(raw, dtype=torch.float32))

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_alpha)

    def forward(self, f0: torch.Tensor, l_a3: torch.Tensor) -> torch.Tensor:
        h, w = f0.shape[-2:]
        l_a3_up = upsample_to(l_a3, (h, w))
        z = torch.cat([self.f0_proj(f0), l_a3_up], dim=1)
        residual = self.final(self.fuse(z))
        return l_a3_up + self.alpha * residual
