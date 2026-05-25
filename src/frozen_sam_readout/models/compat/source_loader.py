"""Load source-repo checkpoint files into source-architecture-compatible models.

Source checkpoints (from texture-representations repo) store weights in
``head_state_dict`` with a different key layout and architecture than the
omniSAM migrated heads:

  Source A0:
    projections.fpn_2.{weight,bias}
    decoder.{0,1,3,4}.{weight,bias}  (Conv2d(bias=True) + GroupNorm, x2)
    classifier.{weight,bias}

  Source A3:
    memory_tokens
    primary_projection.{weight,bias}
    skip_projections.fpn_1.{weight,bias}
    blocks.0.{query_norm,memory_norm,attn.*}
    decoder.{0,1,3,4}.{weight,bias}
    classifier.{weight,bias}

  Source final_staged:
    (all of A3 keys) + raw_alpha + f0_projection + f0_decoder

Because the source uses Conv2d(bias=True) while omniSAM uses Conv2d(bias=False),
direct weight remapping would silently drop the conv biases.  This module
instead reconstructs the exact source architectures and loads weights verbatim,
so inference is provably equivalent to the source repo.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_GN_GROUPS = 32
_DECODER_DIM = 128


# ──────────────────────────── Source building blocks ──────────────────────── #

class _ConvGNReLU(nn.Module):
    """Conv2d(bias=True) + GroupNorm + ReLU — exact source block."""
    def __init__(self, in_ch: int, out_ch: int, k: int = 3) -> None:
        super().__init__()
        # Source code has bias=True (conv weights include bias term)
        self.conv = nn.Conv2d(in_ch, out_ch, k, padding=k // 2, bias=True)
        self.gn   = nn.GroupNorm(_GN_GROUPS, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.gn(self.conv(x)))


class _SourceA0Head(nn.Module):
    """Source fpn_2-only head.  Key layout: projections.fpn_2 / decoder / classifier."""
    def __init__(self) -> None:
        super().__init__()
        self.projections = nn.ModuleDict({"fpn_2": nn.Conv2d(256, _DECODER_DIM, 1)})
        self.decoder = nn.Sequential(
            nn.Conv2d(_DECODER_DIM, _DECODER_DIM, 3, padding=1, bias=True),
            nn.GroupNorm(_GN_GROUPS, _DECODER_DIM),
            nn.ReLU(inplace=True),
            nn.Conv2d(_DECODER_DIM, _DECODER_DIM, 3, padding=1, bias=True),
            nn.GroupNorm(_GN_GROUPS, _DECODER_DIM),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(_DECODER_DIM, 1, 1)

    def forward(self, pyramid: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.classifier(self.decoder(self.projections["fpn_2"](pyramid["fpn_2"])))


class _SourceA3Head(nn.Module):
    """Source A3 head.  primary_projection / skip_projections / blocks / decoder / classifier.

    Forward:
      1. primary_projection(fpn_2): 256→128 at coarse spatial size
      2. memory_attention: 128→128
      3. skip_projections["fpn_1"](fpn_1): 256→128 at fine spatial, downsample to coarse
      4. cat([attended, skip_down], dim=1): 256 channels
      5. decoder: 256→128→128; classifier: 128→1 logit
    """
    def __init__(self, memory_tokens: int = 8, num_heads: int = 4) -> None:
        super().__init__()
        d = _DECODER_DIM
        self.primary_projection = nn.Conv2d(256, d, 1)
        self.skip_projections = nn.ModuleDict({"fpn_1": nn.Conv2d(256, d, 1)})
        self.memory_tokens = nn.Parameter(torch.empty(memory_tokens, d))
        self.blocks = nn.ModuleList([
            _SourceMemBlock(d, num_heads=num_heads),
        ])
        # Decoder takes 2*d channels (cat of primary + skip)
        self.decoder = nn.Sequential(
            nn.Conv2d(d * 2, d, 3, padding=1, bias=True),
            nn.GroupNorm(_GN_GROUPS, d),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, d, 3, padding=1, bias=True),
            nn.GroupNorm(_GN_GROUPS, d),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(d, 1, 1)

    def forward(self, pyramid: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.primary_projection(pyramid["fpn_2"])
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        mem = self.memory_tokens.unsqueeze(0).expand(b, -1, -1)
        for blk in self.blocks:
            tokens = blk(tokens, mem)
        attended = tokens.transpose(1, 2).reshape(b, c, h, w)
        skip = self.skip_projections["fpn_1"](pyramid["fpn_1"])
        skip_down = F.interpolate(skip, size=(h, w), mode="bilinear", align_corners=False)
        fused = torch.cat([attended, skip_down], dim=1)  # [B, 2*d, H, W]
        return self.classifier(self.decoder(fused))


class _SourceA2Head(nn.Module):
    """Source A2 head: coarse_head (A0) + fine_projection + residual_decoder + residual_scale.

    Key layout:
      residual_scale
      coarse_head.projections.fpn_2.{weight,bias}
      coarse_head.decoder.{0,1,3,4}.{weight,bias}
      coarse_head.classifier.{weight,bias}
      fine_projection.{weight,bias}
      residual_decoder.{0,1,3,4}.{weight,bias}
      residual_classifier.{weight,bias}

    Note: the residual branch may use a smaller dim (e.g. 32) than the coarse branch (128).
    Pass fine_dim to construct correctly, or use from_hsd() to auto-detect from checkpoint weights.
    """
    def __init__(self, fine_dim: int = 32) -> None:
        super().__init__()
        d = fine_dim
        gn = min(_GN_GROUPS, d)
        self.coarse_head = _SourceA0Head()
        self.fine_projection = nn.Conv2d(256, d, 1)
        self.residual_decoder = nn.Sequential(
            nn.Conv2d(d, d, 3, padding=1, bias=True),
            nn.GroupNorm(gn, d), nn.ReLU(inplace=True),
            nn.Conv2d(d, d, 3, padding=1, bias=True),
            nn.GroupNorm(gn, d), nn.ReLU(inplace=True),
        )
        self.residual_classifier = nn.Conv2d(d, 1, 1)
        self.residual_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, pyramid: Dict[str, torch.Tensor]) -> torch.Tensor:
        coarse = self.coarse_head(pyramid)
        fine = self.fine_projection(pyramid["fpn_1"])
        res = self.residual_classifier(self.residual_decoder(fine))
        res_up = F.interpolate(res, size=coarse.shape[-2:], mode="bilinear", align_corners=False)
        return coarse + torch.sigmoid(self.residual_scale) * res_up


class _SourceMemBlock(nn.Module):
    def __init__(self, d: int, num_heads: int) -> None:
        super().__init__()
        self.query_norm  = nn.LayerNorm(d)
        self.memory_norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, num_heads=num_heads, batch_first=True)

    def forward(self, tokens: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        q = self.query_norm(tokens)
        k = v = self.memory_norm(memory)
        out, _ = self.attn(q, k, v)
        return tokens + out


class _SourceFinalStagedHead(nn.Module):
    """Source staged final head.  A3 + f0_projection/f0_decoder + raw_alpha.

    Architecture (inferred from head_state_dict weight shapes):
      - Decoder: Conv2d(256, 128) + GN + ReLU + Conv2d(128, 128) + GN + ReLU [takes concat of primary+skip]
      - skip_projection: Conv2d(256, 128) [no ModuleDict, unlike A3]
      - f0_decoder: Conv2d(128, 64) + GN + ReLU + Conv2d(64, 1)  [4 entries at indices 0,1,3]
    """
    def __init__(self, memory_tokens: int = 8, num_heads: int = 4) -> None:
        super().__init__()
        d = _DECODER_DIM
        self.primary_projection = nn.Conv2d(256, d, 1)
        self.skip_projection = nn.Conv2d(256, d, 1)
        self.memory_tokens = nn.Parameter(torch.empty(memory_tokens, d))
        self.blocks = nn.ModuleList([_SourceMemBlock(d, num_heads=num_heads)])
        # Decoder takes concat(primary_attended, skip_down) = 2*d channels
        self.decoder = nn.Sequential(
            nn.Conv2d(d * 2, d, 3, padding=1, bias=True),
            nn.GroupNorm(_GN_GROUPS, d), nn.ReLU(inplace=True),
            nn.Conv2d(d, d, 3, padding=1, bias=True),
            nn.GroupNorm(_GN_GROUPS, d), nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(d, 1, 1)
        self.raw_alpha = nn.Parameter(torch.tensor(0.0))
        self.f0_projection = nn.Conv2d(256, d, 1)
        # f0_decoder: Conv(128→64) + GN(64) + ReLU + Conv(64→1)
        self.f0_decoder = nn.Sequential(
            nn.Conv2d(d, d // 2, 3, padding=1, bias=True),
            nn.GroupNorm(min(_GN_GROUPS, d // 2), d // 2), nn.ReLU(inplace=True),
            nn.Conv2d(d // 2, 1, 1),
        )

    def forward(self, pyramid: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.primary_projection(pyramid["fpn_2"])
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        mem = self.memory_tokens.unsqueeze(0).expand(b, -1, -1)
        for blk in self.blocks:
            tokens = blk(tokens, mem)
        attended = tokens.transpose(1, 2).reshape(b, c, h, w)
        skip = self.skip_projection(pyramid["fpn_1"])
        skip_down = F.interpolate(skip, size=(h, w), mode="bilinear", align_corners=False)
        fused = torch.cat([attended, skip_down], dim=1)
        a3 = self.classifier(self.decoder(fused))
        alpha = torch.sigmoid(self.raw_alpha)
        f0 = self.f0_decoder(self.f0_projection(pyramid["fpn_0"]))
        f0_up = F.interpolate(f0, size=a3.shape[-2:], mode="bilinear", align_corners=False)
        return a3 + alpha * f0_up


# ──────────────────────────── Public loader ───────────────────────────────── #

def load_source_checkpoint_head(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> nn.Module:
    """Load a source-repo checkpoint and return a ready-to-eval nn.Module.

    The returned model is already in ``eval()`` mode.  Use it as:

        model = load_source_checkpoint_head(path, device=device)
        logits = model(pyramid)   # pyramid: dict[str, Tensor], fpn_2/fpn_1/fpn_0

    Args:
        checkpoint_path: Path to ``checkpoint.pt`` from the source repo.
        device: Target device.

    Returns:
        An ``nn.Module`` whose architecture and weights exactly match the
        source checkpoint.

    Raises:
        KeyError: If the checkpoint variant is not recognised.
        RuntimeError: If ``load_state_dict`` fails (shape mismatch).
    """
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    variant: str = str(ckpt.get("variant", ""))
    hsd: Dict[str, Any] = ckpt.get("head_state_dict", {})
    mem_tok: int = int(ckpt.get("memory_token_count", 8))
    attn_heads: int = int(ckpt.get("attention_heads", 4))

    # Infer architecture from checkpoint keys
    has_f0      = "f0_projection.weight" in hsd
    has_mem     = "memory_tokens" in hsd
    has_coarse  = "coarse_head.projections.fpn_2.weight" in hsd  # A2 wrapper

    if has_f0 and has_mem:
        model: nn.Module = _SourceFinalStagedHead(memory_tokens=mem_tok, num_heads=attn_heads)
        logger.info("Detected source architecture: final_staged")
    elif has_mem:
        model = _SourceA3Head(memory_tokens=mem_tok, num_heads=attn_heads)
        logger.info("Detected source architecture: A3")
    elif has_coarse:
        # Read fine_dim from checkpoint weights (residual branch may use smaller dim than coarse)
        fine_dim = int(hsd["fine_projection.weight"].shape[0])
        model = _SourceA2Head(fine_dim=fine_dim)
        logger.info("Detected source architecture: A2 (coarse+fine residual, fine_dim=%d)", fine_dim)
    else:
        model = _SourceA0Head()
        logger.info("Detected source architecture: A0 (fpn_2 only)")

    missing, unexpected = model.load_state_dict(hsd, strict=True)
    if missing:
        logger.warning("Missing keys: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys: %s", unexpected)

    return model.to(device).eval()
