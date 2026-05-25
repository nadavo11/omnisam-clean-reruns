"""A2: fpn_2 coarse readout + fpn_1 residual refine."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..blocks.fpn_refine import FpnRefineBlock
from .a0_fpn2 import A0Fpn2Head


class A2Fpn2Fpn1RefineHead(nn.Module):
    def __init__(
        self,
        f2_channels: int,
        f1_channels: int,
        decoder_dim: int = 128,
    ) -> None:
        super().__init__()
        self.coarse = A0Fpn2Head(f2_channels, decoder_dim=decoder_dim)
        self.refine = FpnRefineBlock(f1_channels, decoder_dim=decoder_dim)

    def forward(self, pyramid: dict[str, torch.Tensor]) -> torch.Tensor:
        coarse_logits = self.coarse(pyramid)
        return self.refine(pyramid["fpn_1"], coarse_logits)
