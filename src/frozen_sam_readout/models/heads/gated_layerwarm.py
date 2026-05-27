"""Gated layer-warmup heads for the MoNuSeg 80-epoch ablation batch.

All extra branches start as strict no-ops (gate=0) and ramp to fully active
(gate=1) according to a deterministic schedule set externally by the training
loop via ``set_gate(name, value)``.

Gate contract:
  • At gate=0: the branch contributes exactly zero to the output.
  • At gate=1: the branch is fully active.
  • Ramp is linear; the training loop is responsible for the schedule.
  • Gates are registered as non-trainable buffers (deterministic schedule,
    not learned).

Two classes:

``GatedWarmupHead``  (variant: gated_a3_f0_warmup)
    Standard A3-style path (fpn_2 coarse + fpn_1 refine) with:
      – memory attention at fpn_2 gated by ``gate_memory``
      – F0 residual branch gated by ``gate_f0``
    Handles variants 1, 2, 4, 5, 6 of the 80-epoch ablation.

``GatedAllRefineHead``  (variant: gated_all_refine_mem)
    Memory attention at every refinement level, each gated independently:
      – ``gate_mem_coarse``  memory at fpn_2 coarse readout
      – ``gate_mem_f1``      memory at fpn_1 refinement step
      – ``gate_f0``          F0 residual branch (also controls F0-level memory)
    Handles variant 3 (all-refine-memory).
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn

from ..blocks.conv_blocks import ConvBNReLU
from ..blocks.fpn_refine import FpnRefineBlock
from ..blocks.memory_attention import MemoryAttentionBlock
from ..blocks.upsample import upsample_to


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _gate_log(gates: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Return a plain-float copy of gate buffer values for logging."""
    return {k: float(v.item()) for k, v in gates.items()}


# ---------------------------------------------------------------------------
# GatedWarmupHead
# ---------------------------------------------------------------------------

class GatedWarmupHead(nn.Module):
    """A3-style head (fpn_2 coarse + fpn_1 refine) with gated memory and F0.

    Both extra branches (memory attention, F0 residual) are strict no-ops when
    their respective gates are 0.  The training loop advances gates each epoch
    by calling ``set_gate(name, value)``.

    Architecture (forward pass):
        x_base = proj(fpn_2)                         # coarse projection
        x      = x_base + gate_memory * mem_delta    # optionally memory-enhanced
        feats  = body(x)                             # 2× ConvBNReLU
        coarse = coarse_head(feats)                  # 1-channel coarse logits
        l_a3   = refine(fpn_1, coarse)               # F1 residual refinement
        if gate_f0 > 0:
            l_up  = upsample(l_a3, fpn_0.shape)
            delta = f0_residual(fpn_0, l_a3) − l_up  # α * R (≈0 at init)
            return l_up + gate_f0 * delta
        return l_a3
    """

    def __init__(
        self,
        f2_channels: int,
        f1_channels: int,
        f0_channels: int,
        decoder_dim: int = 128,
        memory_tokens: int = 4,
        alpha_init: float = 0.05,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(f2_channels, decoder_dim, kernel_size=1)
        self.memory = MemoryAttentionBlock(
            embed_dim=decoder_dim,
            num_memory_tokens=memory_tokens,
        )
        self.body = nn.Sequential(
            ConvBNReLU(decoder_dim, decoder_dim),
            ConvBNReLU(decoder_dim, decoder_dim),
        )
        self.coarse_head = nn.Conv2d(decoder_dim, 1, kernel_size=1)
        self.refine = FpnRefineBlock(f1_channels, decoder_dim=decoder_dim)

        # F0 residual sub-components (mirrors LearnedAlphaResidual but exposed)
        self.f0_proj = nn.Conv2d(f0_channels, decoder_dim, kernel_size=1)
        self.f0_fuse = nn.Sequential(
            ConvBNReLU(decoder_dim + 1, decoder_dim),
            ConvBNReLU(decoder_dim, decoder_dim),
        )
        self.f0_final = nn.Conv2d(decoder_dim, 1, kernel_size=1)
        nn.init.zeros_(self.f0_final.weight)
        nn.init.zeros_(self.f0_final.bias)
        alpha_clamped = float(min(max(alpha_init, 1e-4), 1.0 - 1e-4))
        self.raw_alpha_f0 = nn.Parameter(
            torch.tensor(math.log(alpha_clamped / (1.0 - alpha_clamped)))
        )

        # Deterministic gate buffers — advanced by training loop
        self.register_buffer("gate_memory", torch.tensor(0.0))
        self.register_buffer("gate_f0", torch.tensor(0.0))

    # ------------------------------------------------------------------
    def set_gate(self, name: str, value: float) -> None:
        buf = getattr(self, name, None)
        if buf is None or not isinstance(buf, torch.Tensor):
            raise KeyError(f"No gate buffer named '{name}' on {type(self).__name__}")
        buf.fill_(float(value))

    def gate_state(self) -> Dict[str, float]:
        return {
            "gate_memory": float(self.gate_memory),
            "gate_f0": float(self.gate_f0),
        }

    @property
    def alpha_f0(self) -> float:
        return float(torch.sigmoid(self.raw_alpha_f0).item())

    # ------------------------------------------------------------------
    def forward(self, pyramid: dict[str, torch.Tensor]) -> torch.Tensor:
        x_base = self.proj(pyramid["fpn_2"])

        gate_mem = float(self.gate_memory)
        if gate_mem > 0.0:
            x_with_mem = self.memory(x_base)          # x_base + memory_delta
            mem_delta = x_with_mem - x_base
            x = x_base + self.gate_memory * mem_delta
        else:
            x = x_base

        features = self.body(x)
        coarse_logits = self.coarse_head(features)
        l_a3 = self.refine(pyramid["fpn_1"], coarse_logits)

        gate_f0 = float(self.gate_f0)
        if gate_f0 > 0.0 and "fpn_0" in pyramid:
            fpn0 = pyramid["fpn_0"]
            h0, w0 = fpn0.shape[-2:]
            l_a3_up = upsample_to(l_a3, (h0, w0))
            z = torch.cat([self.f0_proj(fpn0), l_a3_up], dim=1)
            residual = self.f0_final(self.f0_fuse(z))
            alpha = torch.sigmoid(self.raw_alpha_f0)
            f0_delta = alpha * residual               # ≈0 at init (zero-init final)
            return l_a3_up + self.gate_f0 * f0_delta

        return l_a3


# ---------------------------------------------------------------------------
# GatedAllRefineHead
# ---------------------------------------------------------------------------

class GatedAllRefineHead(nn.Module):
    """Memory attention at every refinement scale, each independently gated.

    Gates:
      ``gate_mem_coarse``  – memory at fpn_2 coarse readout
      ``gate_mem_f1``      – memory at fpn_1 mid-scale refinement features
      ``gate_f0``          – F0 residual branch (also controls F0-level memory)

    Architecture:
        x_base  = proj(fpn_2)
        x       = x_base + gate_mem_coarse * mem_delta_coarse
        feats   = body(x)
        coarse  = coarse_head(feats)

        # F1 refinement
        f1_feat = f1_proj(fpn_1)
        f1_feat = f1_feat + gate_mem_f1 * (memory_f1(f1_feat) − f1_feat)
        z_f1    = cat([f1_feat, upsample(coarse)])
        l_a3    = f1_head(f1_fuse(z_f1))

        # F0 residual with F0-level memory
        if gate_f0 > 0:
            l_up    = upsample(l_a3, fpn_0.shape)
            f0_feat = f0_proj(fpn_0)
            f0_feat = f0_feat + gate_f0 * (memory_f0(f0_feat) − f0_feat)  # F0 mem
            z_f0    = cat([f0_feat, l_up])
            residual = f0_final(f0_fuse(z_f0))
            return l_up + gate_f0 * alpha * residual
        return l_a3
    """

    def __init__(
        self,
        f2_channels: int,
        f1_channels: int,
        f0_channels: int,
        decoder_dim: int = 128,
        memory_tokens: int = 4,
        alpha_init: float = 0.05,
    ) -> None:
        super().__init__()
        # Coarse path (fpn_2)
        self.proj = nn.Conv2d(f2_channels, decoder_dim, kernel_size=1)
        self.memory_coarse = MemoryAttentionBlock(
            embed_dim=decoder_dim,
            num_memory_tokens=memory_tokens,
        )
        self.body = nn.Sequential(
            ConvBNReLU(decoder_dim, decoder_dim),
            ConvBNReLU(decoder_dim, decoder_dim),
        )
        self.coarse_head = nn.Conv2d(decoder_dim, 1, kernel_size=1)

        # F1 mid-scale refinement path
        self.f1_proj = nn.Conv2d(f1_channels, decoder_dim, kernel_size=1)
        self.memory_f1 = MemoryAttentionBlock(
            embed_dim=decoder_dim,
            num_memory_tokens=memory_tokens,
        )
        self.f1_fuse = nn.Sequential(
            ConvBNReLU(decoder_dim + 1, decoder_dim),
            ConvBNReLU(decoder_dim, decoder_dim),
        )
        self.f1_head = nn.Conv2d(decoder_dim, 1, kernel_size=1)

        # F0 fine-scale residual path
        self.f0_proj = nn.Conv2d(f0_channels, decoder_dim, kernel_size=1)
        self.memory_f0 = MemoryAttentionBlock(
            embed_dim=decoder_dim,
            num_memory_tokens=memory_tokens,
        )
        self.f0_fuse = nn.Sequential(
            ConvBNReLU(decoder_dim + 1, decoder_dim),
            ConvBNReLU(decoder_dim, decoder_dim),
        )
        self.f0_final = nn.Conv2d(decoder_dim, 1, kernel_size=1)
        nn.init.zeros_(self.f0_final.weight)
        nn.init.zeros_(self.f0_final.bias)
        alpha_clamped = float(min(max(alpha_init, 1e-4), 1.0 - 1e-4))
        self.raw_alpha_f0 = nn.Parameter(
            torch.tensor(math.log(alpha_clamped / (1.0 - alpha_clamped)))
        )

        # Deterministic gate buffers
        self.register_buffer("gate_mem_coarse", torch.tensor(0.0))
        self.register_buffer("gate_mem_f1", torch.tensor(0.0))
        self.register_buffer("gate_f0", torch.tensor(0.0))

    # ------------------------------------------------------------------
    def set_gate(self, name: str, value: float) -> None:
        buf = getattr(self, name, None)
        if buf is None or not isinstance(buf, torch.Tensor):
            raise KeyError(f"No gate buffer named '{name}' on {type(self).__name__}")
        buf.fill_(float(value))

    def gate_state(self) -> Dict[str, float]:
        return {
            "gate_mem_coarse": float(self.gate_mem_coarse),
            "gate_mem_f1": float(self.gate_mem_f1),
            "gate_f0": float(self.gate_f0),
        }

    @property
    def alpha_f0(self) -> float:
        return float(torch.sigmoid(self.raw_alpha_f0).item())

    # ------------------------------------------------------------------
    def forward(self, pyramid: dict[str, torch.Tensor]) -> torch.Tensor:
        # Coarse path with optional memory
        x_base = self.proj(pyramid["fpn_2"])
        gate_mc = float(self.gate_mem_coarse)
        if gate_mc > 0.0:
            x_mem = self.memory_coarse(x_base)
            x = x_base + self.gate_mem_coarse * (x_mem - x_base)
        else:
            x = x_base
        features = self.body(x)
        coarse_logits = self.coarse_head(features)

        # F1 refinement with optional memory
        fpn1 = pyramid["fpn_1"]
        h1, w1 = fpn1.shape[-2:]
        coarse_up = upsample_to(coarse_logits, (h1, w1))
        f1_feat = self.f1_proj(fpn1)
        gate_mf1 = float(self.gate_mem_f1)
        if gate_mf1 > 0.0:
            f1_mem = self.memory_f1(f1_feat)
            f1_feat = f1_feat + self.gate_mem_f1 * (f1_mem - f1_feat)
        z_f1 = torch.cat([f1_feat, coarse_up], dim=1)
        l_a3 = self.f1_head(self.f1_fuse(z_f1))

        # F0 residual with F0-level memory (both controlled by gate_f0)
        gate_f0 = float(self.gate_f0)
        if gate_f0 > 0.0 and "fpn_0" in pyramid:
            fpn0 = pyramid["fpn_0"]
            h0, w0 = fpn0.shape[-2:]
            l_a3_up = upsample_to(l_a3, (h0, w0))
            f0_feat = self.f0_proj(fpn0)
            f0_mem = self.memory_f0(f0_feat)
            f0_feat = f0_feat + self.gate_f0 * (f0_mem - f0_feat)
            z_f0 = torch.cat([f0_feat, l_a3_up], dim=1)
            residual = self.f0_final(self.f0_fuse(z_f0))
            alpha = torch.sigmoid(self.raw_alpha_f0)
            return l_a3_up + self.gate_f0 * alpha * residual

        return l_a3
