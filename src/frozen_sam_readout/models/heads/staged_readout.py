"""Staged Multiscale Frozen-SAM3 Readout: the official final method head.

Stage 1: train ``A3`` (fpn_2 coarse + fpn_1 residual + M=8 memory attention) to
produce ``L_A3`` for 40 epochs (MoNuSeg) or 200 epochs (GlaS).

Stage 2: add a learned-alpha F0 residual head ``R0 = F0Residual(F0, up(L_A3))``;
``L_final = up(L_A3) + alpha * R0`` with ``alpha = sigmoid(raw_alpha)`` and
``raw_alpha`` initialized so ``alpha ≈ 0.05``. The final 1x1 conv of the F0
residual head is zero-initialized so ``L_final ≈ up(L_A3)`` at the start of
stage-2 training. We then jointly train both stages for +40 epochs (do NOT
freeze A3 in the default protocol).

Prediction = ``sigmoid(L_final) > 0.5``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..blocks.learned_alpha import LearnedAlphaResidual
from .a2_fpn2_fpn1_refine import A2Fpn2Fpn1RefineHead
from .a3_memory import A3MemoryHead


class StagedMultiscaleReadoutHead(nn.Module):
    def __init__(
        self,
        f2_channels: int,
        f1_channels: int,
        f0_channels: int,
        decoder_dim: int = 128,
        projection_dim: int = 128,  # accepted for spec parity
        memory_tokens: int = 8,
        alpha_init: float = 0.05,
        final_layer_zero_init: bool = True,
    ) -> None:
        super().__init__()
        del projection_dim  # unused; surface kept for config parity
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
            final_layer_zero_init=final_layer_zero_init,
        )
        self._f0_enabled = True

    def set_f0_enabled(self, enabled: bool) -> None:
        """Enable/disable the F0 residual branch (used during stage-1 training)."""

        self._f0_enabled = bool(enabled)

    def freeze_a3(self, freeze: bool = True) -> None:
        """Optional helper used by the freeze-A3 ablation control."""

        for p in self.a3.parameters():
            p.requires_grad_(not freeze)

    @property
    def alpha(self) -> Optional[torch.Tensor]:
        return self.f0_residual.alpha if self._f0_enabled else None

    def forward(self, pyramid: dict[str, torch.Tensor]) -> torch.Tensor:
        l_a3 = self.a3(pyramid)
        if not self._f0_enabled or "fpn_0" not in pyramid:
            return l_a3
        return self.f0_residual(pyramid["fpn_0"], l_a3)


class StagedA2ReadoutHead(nn.Module):
    """Staged variant without memory attention: A2 (fpn_2+fpn_1 refine) + F0 residual.

    Stage 1: train A2 (coarse fpn_2 readout + fpn_1 residual refine, no memory).
    Stage 2: add learned-alpha F0 residual on top, train jointly.

    Ablation control: tests whether memory attention in the staged protocol
    contributes beyond the A2+F0 baseline.
    """

    def __init__(
        self,
        f2_channels: int,
        f1_channels: int,
        f0_channels: int,
        decoder_dim: int = 128,
        alpha_init: float = 0.05,
        final_layer_zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.a2 = A2Fpn2Fpn1RefineHead(
            f2_channels=f2_channels,
            f1_channels=f1_channels,
            decoder_dim=decoder_dim,
        )
        self.f0_residual = LearnedAlphaResidual(
            f0_channels=f0_channels,
            decoder_dim=decoder_dim,
            alpha_init=alpha_init,
            final_layer_zero_init=final_layer_zero_init,
        )
        self._f0_enabled = True

    def set_f0_enabled(self, enabled: bool) -> None:
        self._f0_enabled = bool(enabled)

    @property
    def alpha(self) -> Optional[torch.Tensor]:
        return self.f0_residual.alpha if self._f0_enabled else None

    def forward(self, pyramid: dict[str, torch.Tensor]) -> torch.Tensor:
        l_a2 = self.a2(pyramid)
        if not self._f0_enabled or "fpn_0" not in pyramid:
            return l_a2
        return self.f0_residual(pyramid["fpn_0"], l_a2)
