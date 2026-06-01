"""Readout heads for SAM decoder dense semantic maps."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..blocks.conv_blocks import ConvBNReLU
from .a3_memory import A3MemoryHead


class DecoderSemanticLinearHead(nn.Module):
    """Minimal 1x1 linear probe on the SAM decoder semantic map."""

    def __init__(self, decoder_map_channels: int, decoder_dim: int = 128) -> None:
        super().__init__()
        del decoder_dim
        self.head = nn.Conv2d(decoder_map_channels, 1, kernel_size=1)

    def forward(self, pyramid: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.head(pyramid["decoder_semantic_map"])


class DecoderSemanticSmallHead(nn.Module):
    """Shallow practical head over the SAM decoder semantic map."""

    def __init__(self, decoder_map_channels: int, decoder_dim: int = 128) -> None:
        super().__init__()
        self.proj = nn.Conv2d(decoder_map_channels, decoder_dim, kernel_size=1)
        self.body = nn.Sequential(
            ConvBNReLU(decoder_dim, decoder_dim),
            ConvBNReLU(decoder_dim, decoder_dim),
        )
        self.head = nn.Conv2d(decoder_dim, 1, kernel_size=1)

    def forward(self, pyramid: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.head(self.body(self.proj(pyramid["decoder_semantic_map"])))


class FpnPlusDecoderSemanticHead(nn.Module):
    """Fuse the current A3-style FPN/memory path with a decoder-map residual."""

    def __init__(
        self,
        f2_channels: int,
        f1_channels: int,
        decoder_map_channels: int,
        decoder_dim: int = 128,
        memory_tokens: int = 4,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.fpn = A3MemoryHead(
            f2_channels=f2_channels,
            f1_channels=f1_channels,
            decoder_dim=decoder_dim,
            memory_tokens=memory_tokens,
            num_heads=num_heads,
        )
        self.decoder = DecoderSemanticSmallHead(
            decoder_map_channels=decoder_map_channels,
            decoder_dim=decoder_dim,
        )

    def forward(self, pyramid: dict[str, torch.Tensor]) -> torch.Tensor:
        decoder_logits = self.decoder(pyramid)
        fpn_logits = self.fpn(pyramid)
        decoder_logits = F.interpolate(
            decoder_logits,
            size=fpn_logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return fpn_logits + decoder_logits


class DecoderSemanticFpnFusionHead(nn.Module):
    """Concatenate projected D0 + FPN levels at the decoder-map resolution."""

    def __init__(
        self,
        f2_channels: int,
        f1_channels: int,
        f0_channels: int,
        decoder_map_channels: int,
        decoder_dim: int = 128,
        projection_dim: int = 64,
    ) -> None:
        super().__init__()
        self.projections = nn.ModuleDict(
            {
                "decoder_semantic_map": nn.Conv2d(decoder_map_channels, projection_dim, 1),
                "fpn_2": nn.Conv2d(f2_channels, projection_dim, 1),
                "fpn_1": nn.Conv2d(f1_channels, projection_dim, 1),
                "fpn_0": nn.Conv2d(f0_channels, projection_dim, 1),
            }
        )
        in_channels = projection_dim * len(self.projections)
        self.body = nn.Sequential(
            ConvBNReLU(in_channels, decoder_dim),
            ConvBNReLU(decoder_dim, decoder_dim),
            ConvBNReLU(decoder_dim, decoder_dim),
        )
        self.head = nn.Conv2d(decoder_dim, 1, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, pyramid: dict[str, torch.Tensor]) -> torch.Tensor:
        target_hw = pyramid["decoder_semantic_map"].shape[-2:]
        parts = []
        for key, proj in self.projections.items():
            x = proj(pyramid[key])
            if x.shape[-2:] != target_hw:
                x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
            parts.append(x)
        return self.head(self.body(torch.cat(parts, dim=1)))


class DecoderSemanticF0FusionHead(nn.Module):
    """Concatenate projected D0 + fine F0 features at decoder-map resolution."""

    def __init__(
        self,
        f0_channels: int,
        decoder_map_channels: int,
        decoder_dim: int = 128,
        projection_dim: int = 64,
        **_: int,
    ) -> None:
        super().__init__()
        self.projections = nn.ModuleDict(
            {
                "decoder_semantic_map": nn.Conv2d(decoder_map_channels, projection_dim, 1),
                "fpn_0": nn.Conv2d(f0_channels, projection_dim, 1),
            }
        )
        self.body = nn.Sequential(
            ConvBNReLU(projection_dim * 2, decoder_dim),
            ConvBNReLU(decoder_dim, decoder_dim),
            ConvBNReLU(decoder_dim, decoder_dim),
        )
        self.head = nn.Conv2d(decoder_dim, 1, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, pyramid: dict[str, torch.Tensor]) -> torch.Tensor:
        target_hw = pyramid["decoder_semantic_map"].shape[-2:]
        parts = []
        for key, proj in self.projections.items():
            x = proj(pyramid[key])
            if x.shape[-2:] != target_hw:
                x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
            parts.append(x)
        return self.head(self.body(torch.cat(parts, dim=1)))


class DecoderSemanticMultiFusionHead(nn.Module):
    """Generic D0 + arbitrary FPN-level concatenation at decoder-map resolution."""

    def __init__(
        self,
        *,
        source_channels: dict[str, int],
        source_keys: tuple[str, ...],
        decoder_dim: int = 128,
        projection_dim: int = 64,
    ) -> None:
        super().__init__()
        if not source_keys:
            raise ValueError("source_keys must not be empty")
        self.source_keys = tuple(source_keys)
        self.projections = nn.ModuleDict(
            {key: nn.Conv2d(int(source_channels[key]), projection_dim, 1) for key in self.source_keys}
        )
        in_channels = projection_dim * len(self.source_keys)
        self.body = nn.Sequential(
            ConvBNReLU(in_channels, decoder_dim),
            ConvBNReLU(decoder_dim, decoder_dim),
            ConvBNReLU(decoder_dim, decoder_dim),
        )
        self.head = nn.Conv2d(decoder_dim, 1, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, pyramid: dict[str, torch.Tensor]) -> torch.Tensor:
        target_hw = pyramid["decoder_semantic_map"].shape[-2:]
        parts = []
        for key in self.source_keys:
            x = self.projections[key](pyramid[key])
            if x.shape[-2:] != target_hw:
                x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
            parts.append(x)
        return self.head(self.body(torch.cat(parts, dim=1)))
