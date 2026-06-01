"""Lightweight attention/context fusion heads for frozen D0 + FPN readout.

The base pattern is shared across all variants:

* project each frozen source to a common width;
* resize all projected maps to the decoder-map resolution;
* build a baseline fused tensor by concatenation + 1x1 compression;
* optionally inject a residual context delta;
* decode with a small ConvBNReLU stack and a zero-initialized 1x1 head.

All modules are intentionally small and residual-safe at initialization.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..blocks.conv_blocks import ConvBNReLU
from .decoder_semantic_map import DecoderSemanticFpnFusionHead


def _resize_to(x: torch.Tensor, size_hw: tuple[int, int]) -> torch.Tensor:
    if x.shape[-2:] == size_hw:
        return x
    return F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)


def _entropy_from_probs(probs: torch.Tensor, dim: int = -1) -> torch.Tensor:
    probs = probs.clamp_min(1e-8)
    ent = -(probs * probs.log()).sum(dim=dim)
    norm = math.log(float(probs.shape[dim])) if probs.shape[dim] > 1 else 1.0
    return ent / norm


def _pool_tokens(x: torch.Tensor, pool_hw: tuple[int, int]) -> torch.Tensor:
    pooled = F.adaptive_avg_pool2d(x, pool_hw)
    return pooled.flatten(2).transpose(1, 2)


def _tokens_to_map(tokens: torch.Tensor, pool_hw: tuple[int, int]) -> torch.Tensor:
    b, n, c = tokens.shape
    h, w = pool_hw
    if n != h * w:
        raise ValueError(f"token count {n} did not match pool_hw={pool_hw}")
    return tokens.transpose(1, 2).reshape(b, c, h, w)


def _partition_windows(x: torch.Tensor, window_size: int) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    b, c, h, w = x.shape
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    hp, wp = h + pad_h, w + pad_w
    windows = (
        x.view(b, c, hp // window_size, window_size, wp // window_size, window_size)
        .permute(0, 2, 4, 3, 5, 1)
        .reshape(-1, window_size * window_size, c)
    )
    return windows, (h, w, hp, wp)


def _reverse_windows(tokens: torch.Tensor, window_size: int, meta: tuple[int, int, int, int], batch_size: int) -> torch.Tensor:
    h, w, hp, wp = meta
    c = tokens.shape[-1]
    windows_per_image = (hp // window_size) * (wp // window_size)
    x = (
        tokens.reshape(batch_size, windows_per_image, window_size, window_size, c)
        .reshape(batch_size, hp // window_size, wp // window_size, window_size, window_size, c)
        .permute(0, 5, 1, 3, 2, 4)
        .reshape(batch_size, c, hp, wp)
    )
    return x[:, :, :h, :w]


def _zero_init_conv(conv: nn.Conv2d) -> None:
    nn.init.zeros_(conv.weight)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


class _AttentionContextFusionBase(nn.Module):
    """Shared D0+FPN projection/fusion scaffold."""

    attention_type = "base"

    def __init__(
        self,
        *,
        source_channels: dict[str, int],
        source_keys: Sequence[str],
        decoder_dim: int = 128,
        projection_dim: int = 64,
    ) -> None:
        super().__init__()
        if not source_keys:
            raise ValueError("source_keys must not be empty")
        self.source_keys = tuple(source_keys)
        self.decoder_dim = int(decoder_dim)
        self.projection_dim = int(projection_dim)
        self.projections = nn.ModuleDict(
            {key: nn.Conv2d(int(source_channels[key]), self.projection_dim, kernel_size=1) for key in self.source_keys}
        )
        self.pre_fuse = nn.Conv2d(self.projection_dim * len(self.source_keys), self.decoder_dim, kernel_size=1)
        self.body = nn.Sequential(
            ConvBNReLU(self.decoder_dim, self.decoder_dim),
            ConvBNReLU(self.decoder_dim, self.decoder_dim),
            ConvBNReLU(self.decoder_dim, self.decoder_dim),
        )
        self.head = nn.Conv2d(self.decoder_dim, 1, kernel_size=1)
        _zero_init_conv(self.head)
        self._last_context_summary: dict[str, float] = {}
        self._last_attention_maps: dict[str, torch.Tensor] = {}

    def _project_sources(self, pyramid: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], tuple[int, int]]:
        target_hw = pyramid["decoder_semantic_map"].shape[-2:]
        projected: dict[str, torch.Tensor] = {}
        for key in self.source_keys:
            x = self.projections[key](pyramid[key])
            projected[key] = _resize_to(x, target_hw)
        return projected, target_hw

    def _baseline(self, projected: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.pre_fuse(torch.cat([projected[key] for key in self.source_keys], dim=1))

    def _context_delta(
        self,
        projected: dict[str, torch.Tensor],
        baseline: torch.Tensor,
        target_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
        del projected, baseline, target_hw
        return torch.zeros_like(baseline), {}, {}

    def forward(self, pyramid: dict[str, torch.Tensor]) -> torch.Tensor:
        projected, target_hw = self._project_sources(pyramid)
        baseline = self._baseline(projected)
        delta, summary, maps = self._context_delta(projected, baseline, target_hw)
        self._last_context_summary = summary
        self._last_attention_maps = maps
        fused = baseline + delta
        return self.head(self.body(fused))

    def context_state_summary(self) -> dict[str, float]:
        return dict(self._last_context_summary)

    def attention_maps(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in self._last_attention_maps.items()}


class SourceGatedChannelContextHead(_AttentionContextFusionBase):
    attention_type = "source_gated_channel_context"

    def __init__(
        self,
        *,
        source_channels: dict[str, int],
        source_keys: Sequence[str],
        decoder_dim: int = 128,
        projection_dim: int = 64,
        gate_scale: float = 0.1,
    ) -> None:
        super().__init__(
            source_channels=source_channels,
            source_keys=source_keys,
            decoder_dim=decoder_dim,
            projection_dim=projection_dim,
        )
        self.gate_scale = float(gate_scale)
        pooled_dim = self.projection_dim * len(self.source_keys)
        hidden = max(128, pooled_dim // 2)
        self.gate_mlp = nn.Sequential(
            nn.Linear(pooled_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.source_gate_head = nn.Linear(hidden, len(self.source_keys))
        self.channel_gate_head = nn.Linear(hidden, len(self.source_keys) * self.projection_dim)
        nn.init.zeros_(self.source_gate_head.weight)
        nn.init.zeros_(self.source_gate_head.bias)
        nn.init.zeros_(self.channel_gate_head.weight)
        nn.init.zeros_(self.channel_gate_head.bias)

    def _context_delta(self, projected, baseline, target_hw):
        pooled = torch.cat([_pool_tokens(projected[k], (1, 1)).squeeze(1) for k in self.source_keys], dim=1)
        hidden = self.gate_mlp(pooled)
        source_gate = 1.0 + self.gate_scale * torch.tanh(self.source_gate_head(hidden))
        channel_gate = 1.0 + self.gate_scale * torch.tanh(self.channel_gate_head(hidden))
        channel_gate = channel_gate.view(-1, len(self.source_keys), self.projection_dim, 1, 1)
        gated_parts = []
        for i, key in enumerate(self.source_keys):
            gated_parts.append(projected[key] * source_gate[:, i].view(-1, 1, 1, 1) * channel_gate[:, i])
        gated = torch.cat(gated_parts, dim=1)
        delta = self.pre_fuse(gated) - baseline
        summary = {f"source_gate/{key}": float(source_gate[:, i].mean().item()) for i, key in enumerate(self.source_keys)}
        summary.update({
            "source_gate/channel_mean": float(channel_gate.mean().item()),
            "source_gate/source_mean": float(source_gate.mean().item()),
        })
        return delta, summary, {}


class CbamSpatialChannelAttentionFusionHead(_AttentionContextFusionBase):
    attention_type = "spatial_channel_attention_fusion"

    def __init__(
        self,
        *,
        source_channels: dict[str, int],
        source_keys: Sequence[str],
        decoder_dim: int = 128,
        projection_dim: int = 64,
        reduction: int = 4,
    ) -> None:
        super().__init__(
            source_channels=source_channels,
            source_keys=source_keys,
            decoder_dim=decoder_dim,
            projection_dim=projection_dim,
        )
        hidden = max(8, self.decoder_dim // int(reduction))
        self.channel_mlp = nn.Sequential(
            nn.Linear(self.decoder_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.decoder_dim),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        nn.init.zeros_(self.channel_mlp[-1].weight)
        nn.init.zeros_(self.channel_mlp[-1].bias)
        _zero_init_conv(self.spatial)

    def _context_delta(self, projected, baseline, target_hw):
        pooled = baseline.mean(dim=(2, 3)) + baseline.amax(dim=(2, 3))
        channel = 1.0 + 0.1 * torch.tanh(self.channel_mlp(pooled)).view(-1, self.decoder_dim, 1, 1)
        avg = baseline.mean(dim=1, keepdim=True)
        mx = baseline.amax(dim=1, keepdim=True)
        spatial = 1.0 + 0.1 * torch.tanh(self.spatial(torch.cat([avg, mx], dim=1)))
        attention = channel * spatial
        delta = baseline * attention - baseline
        summary = {
            "cbam/channel_mean": float(channel.mean().item()),
            "cbam/spatial_mean": float(spatial.mean().item()),
            "cbam/attention_mean": float(attention.mean().item()),
        }
        return delta, summary, {"cbam_spatial": spatial.detach()}


class _CrossAttentionDelta(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.norm_q = nn.LayerNorm(self.embed_dim)
        self.norm_kv = nn.LayerNorm(self.embed_dim)
        self.attn = nn.MultiheadAttention(self.embed_dim, num_heads=num_heads, batch_first=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, q_tokens: torch.Tensor, kv_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.norm_q(q_tokens)
        kv = self.norm_kv(kv_tokens)
        out, weights = self.attn(q, kv, kv, need_weights=True, average_attn_weights=False)
        delta = self.out_proj(out)
        attn_entropy = float(_entropy_from_probs(weights.mean(dim=1), dim=-1).mean().item())
        return delta, torch.tensor(attn_entropy, device=q_tokens.device)


class D0QueryFpnCrossAttentionHead(_AttentionContextFusionBase):
    attention_type = "decoder_queries_encoder_cross_attention"

    def __init__(
        self,
        *,
        source_channels: dict[str, int],
        source_keys: Sequence[str],
        decoder_dim: int = 128,
        projection_dim: int = 64,
        num_heads: int = 4,
        pool_hw: int = 16,
    ) -> None:
        super().__init__(
            source_channels=source_channels,
            source_keys=source_keys,
            decoder_dim=decoder_dim,
            projection_dim=projection_dim,
        )
        self.pool_hw = int(pool_hw)
        self.context_proj = nn.Conv2d(self.projection_dim * (len(self.source_keys) - 1), self.decoder_dim, kernel_size=1)
        self.query_proj = nn.Conv2d(self.projection_dim, self.decoder_dim, kernel_size=1)
        self.cross = _CrossAttentionDelta(self.decoder_dim, num_heads=num_heads)

    def _context_delta(self, projected, baseline, target_hw):
        d0 = projected["decoder_semantic_map"]
        fpn = [projected[k] for k in self.source_keys if k != "decoder_semantic_map"]
        kv_map = self.context_proj(torch.cat(fpn, dim=1))
        q_tokens = _pool_tokens(self.query_proj(d0), (self.pool_hw, self.pool_hw))
        kv_tokens = _pool_tokens(kv_map, (self.pool_hw, self.pool_hw))
        delta_tokens, entropy = self.cross(q_tokens, kv_tokens)
        delta = _tokens_to_map(delta_tokens, (self.pool_hw, self.pool_hw))
        delta = _resize_to(delta, target_hw)
        summary = {"cross_attention/entropy": float(entropy.item())}
        return delta, summary, {"cross_attention": _resize_to(delta.detach(), target_hw)}


class FpnQueryD0CrossAttentionHead(_AttentionContextFusionBase):
    attention_type = "encoder_queries_decoder_cross_attention"

    def __init__(
        self,
        *,
        source_channels: dict[str, int],
        source_keys: Sequence[str],
        decoder_dim: int = 128,
        projection_dim: int = 64,
        num_heads: int = 4,
        pool_hw: int = 16,
    ) -> None:
        super().__init__(
            source_channels=source_channels,
            source_keys=source_keys,
            decoder_dim=decoder_dim,
            projection_dim=projection_dim,
        )
        self.pool_hw = int(pool_hw)
        self.context_proj = nn.Conv2d(self.projection_dim, self.decoder_dim, kernel_size=1)
        self.query_proj = nn.Conv2d(self.projection_dim * (len(self.source_keys) - 1), self.decoder_dim, kernel_size=1)
        self.cross = _CrossAttentionDelta(self.decoder_dim, num_heads=num_heads)

    def _context_delta(self, projected, baseline, target_hw):
        d0 = projected["decoder_semantic_map"]
        fpn = [projected[k] for k in self.source_keys if k != "decoder_semantic_map"]
        q_map = self.query_proj(torch.cat(fpn, dim=1))
        kv_map = self.context_proj(d0)
        q_tokens = _pool_tokens(q_map, (self.pool_hw, self.pool_hw))
        kv_tokens = _pool_tokens(kv_map, (self.pool_hw, self.pool_hw))
        delta_tokens, entropy = self.cross(q_tokens, kv_tokens)
        delta = _tokens_to_map(delta_tokens, (self.pool_hw, self.pool_hw))
        delta = _resize_to(delta, target_hw)
        summary = {"cross_attention/entropy": float(entropy.item())}
        return delta, summary, {"cross_attention": _resize_to(delta.detach(), target_hw)}


class BidirectionalDecoderEncoderCrossAttentionHead(_AttentionContextFusionBase):
    attention_type = "bidirectional_decoder_encoder_cross_attention"

    def __init__(
        self,
        *,
        source_channels: dict[str, int],
        source_keys: Sequence[str],
        decoder_dim: int = 128,
        projection_dim: int = 64,
        num_heads: int = 4,
        pool_hw: int = 16,
    ) -> None:
        super().__init__(
            source_channels=source_channels,
            source_keys=source_keys,
            decoder_dim=decoder_dim,
            projection_dim=projection_dim,
        )
        self.pool_hw = int(pool_hw)
        self.d0_to_fpn = D0QueryFpnCrossAttentionHead(
            source_channels=source_channels,
            source_keys=source_keys,
            decoder_dim=decoder_dim,
            projection_dim=projection_dim,
            num_heads=num_heads,
            pool_hw=pool_hw,
        )
        self.fpn_to_d0 = FpnQueryD0CrossAttentionHead(
            source_channels=source_channels,
            source_keys=source_keys,
            decoder_dim=decoder_dim,
            projection_dim=projection_dim,
            num_heads=num_heads,
            pool_hw=pool_hw,
        )

    def _context_delta(self, projected, baseline, target_hw):
        delta_a, summary_a, map_a = self.d0_to_fpn._context_delta(projected, baseline, target_hw)
        delta_b, summary_b, map_b = self.fpn_to_d0._context_delta(projected, baseline, target_hw)
        summary = {
            "bidirectional/d0_to_fpn_entropy": summary_a["cross_attention/entropy"],
            "bidirectional/fpn_to_d0_entropy": summary_b["cross_attention/entropy"],
        }
        maps = {"d0_to_fpn": map_a["cross_attention"], "fpn_to_d0": map_b["cross_attention"]}
        return 0.5 * (delta_a + delta_b), summary, maps


class LearnedTaskTokensFiLMHead(_AttentionContextFusionBase):
    attention_type = "learned_task_context_tokens"

    def __init__(
        self,
        *,
        source_channels: dict[str, int],
        source_keys: Sequence[str],
        decoder_dim: int = 128,
        projection_dim: int = 64,
        num_tokens: int = 4,
        num_heads: int = 4,
        pool_hw: int = 16,
    ) -> None:
        super().__init__(
            source_channels=source_channels,
            source_keys=source_keys,
            decoder_dim=decoder_dim,
            projection_dim=projection_dim,
        )
        self.pool_hw = int(pool_hw)
        self.num_tokens = int(num_tokens)
        self.tokens = nn.Parameter(torch.empty(self.num_tokens, self.decoder_dim))
        nn.init.normal_(self.tokens, mean=0.0, std=0.02)
        self.token_attn = nn.MultiheadAttention(self.decoder_dim, num_heads=num_heads, batch_first=True)
        self.token_norm = nn.LayerNorm(self.decoder_dim)
        self.gamma = nn.Linear(self.decoder_dim, self.decoder_dim)
        self.beta = nn.Linear(self.decoder_dim, self.decoder_dim)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def _context_delta(self, projected, baseline, target_hw):
        ctx = _pool_tokens(baseline, (self.pool_hw, self.pool_hw))
        tokens = self.tokens.unsqueeze(0).expand(baseline.shape[0], -1, -1)
        out, weights = self.token_attn(query=self.token_norm(tokens), key=self.token_norm(ctx), value=self.token_norm(ctx), need_weights=True, average_attn_weights=False)
        pooled = out.mean(dim=1)
        gamma = 1.0 + 0.1 * torch.tanh(self.gamma(pooled)).view(-1, self.decoder_dim, 1, 1)
        beta = 0.1 * torch.tanh(self.beta(pooled)).view(-1, self.decoder_dim, 1, 1)
        delta = baseline * gamma + beta - baseline
        summary = {
            "task_tokens/gamma_mean": float(gamma.mean().item()),
            "task_tokens/beta_mean": float(beta.mean().item()),
            "task_tokens/token_norm": float(tokens.norm(dim=-1).mean().item()),
            "task_tokens/attention_entropy": float(_entropy_from_probs(weights.mean(dim=1), dim=-1).mean().item()),
        }
        return delta, summary, {"task_tokens_attention": weights.mean(dim=1).detach()}


class WindowSelfAttentionFusionHead(_AttentionContextFusionBase):
    attention_type = "local_window_self_attention_fusion"

    def __init__(
        self,
        *,
        source_channels: dict[str, int],
        source_keys: Sequence[str],
        decoder_dim: int = 128,
        projection_dim: int = 64,
        num_heads: int = 4,
        pool_hw: int = 16,
        window_size: int = 4,
    ) -> None:
        super().__init__(
            source_channels=source_channels,
            source_keys=source_keys,
            decoder_dim=decoder_dim,
            projection_dim=projection_dim,
        )
        self.pool_hw = int(pool_hw)
        self.window_size = int(window_size)
        self.self_attn = nn.MultiheadAttention(self.decoder_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(self.decoder_dim)
        self.out_proj = nn.Linear(self.decoder_dim, self.decoder_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _context_delta(self, projected, baseline, target_hw):
        pooled = F.adaptive_avg_pool2d(baseline, (self.pool_hw, self.pool_hw))
        windows, meta = _partition_windows(pooled, self.window_size)
        tokens = self.norm(windows)
        out, weights = self.self_attn(tokens, tokens, tokens, need_weights=True, average_attn_weights=False)
        delta_tokens = self.out_proj(out)
        pooled_delta = _reverse_windows(delta_tokens, self.window_size, meta, baseline.shape[0])
        delta = _resize_to(pooled_delta, target_hw)
        summary = {
            "window_self_attention/entropy": float(_entropy_from_probs(weights.mean(dim=1), dim=-1).mean().item()),
            "window_self_attention/window_size": float(self.window_size),
        }
        return delta, summary, {"window_self_attention": _resize_to(pooled_delta.detach(), target_hw)}


class LowRankGlobalContextFusionHead(_AttentionContextFusionBase):
    attention_type = "low_rank_global_context_fusion"

    def __init__(
        self,
        *,
        source_channels: dict[str, int],
        source_keys: Sequence[str],
        decoder_dim: int = 128,
        projection_dim: int = 64,
        num_heads: int = 4,
        query_hw: int = 16,
        context_hw: int = 8,
    ) -> None:
        super().__init__(
            source_channels=source_channels,
            source_keys=source_keys,
            decoder_dim=decoder_dim,
            projection_dim=projection_dim,
        )
        self.query_hw = int(query_hw)
        self.context_hw = int(context_hw)
        self.q_attn = nn.MultiheadAttention(self.decoder_dim, num_heads=num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(self.decoder_dim)
        self.norm_kv = nn.LayerNorm(self.decoder_dim)
        self.out_proj = nn.Linear(self.decoder_dim, self.decoder_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _context_delta(self, projected, baseline, target_hw):
        q_map = F.adaptive_avg_pool2d(baseline, (self.query_hw, self.query_hw))
        kv_map = F.adaptive_avg_pool2d(baseline, (self.context_hw, self.context_hw))
        q = self.norm_q(q_map.flatten(2).transpose(1, 2))
        kv = self.norm_kv(kv_map.flatten(2).transpose(1, 2))
        out, weights = self.q_attn(q, kv, kv, need_weights=True, average_attn_weights=False)
        delta = self.out_proj(out).transpose(1, 2).reshape(-1, self.decoder_dim, self.query_hw, self.query_hw)
        delta = _resize_to(delta, target_hw)
        summary = {
            "global_context/entropy": float(_entropy_from_probs(weights.mean(dim=1), dim=-1).mean().item()),
            "global_context/query_hw": float(self.query_hw),
            "global_context/context_hw": float(self.context_hw),
        }
        return delta, summary, {"global_context_attention": _resize_to(delta.detach(), target_hw)}


class _TaskTokenContextModule(nn.Module):
    """Learned task-token context extractor over a fused feature map."""

    def __init__(self, decoder_dim: int, *, num_tokens: int = 4, num_heads: int = 4, pool_hw: int = 16) -> None:
        super().__init__()
        self.decoder_dim = int(decoder_dim)
        self.num_tokens = int(num_tokens)
        self.pool_hw = int(pool_hw)
        self.tokens = nn.Parameter(torch.empty(self.num_tokens, self.decoder_dim))
        nn.init.normal_(self.tokens, mean=0.0, std=0.02)
        self.token_attn = nn.MultiheadAttention(self.decoder_dim, num_heads=num_heads, batch_first=True)
        self.token_norm = nn.LayerNorm(self.decoder_dim)
        self.context_norm = nn.LayerNorm(self.decoder_dim)

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
        ctx = _pool_tokens(feat, (self.pool_hw, self.pool_hw))
        tokens = self.tokens.unsqueeze(0).expand(feat.shape[0], -1, -1)
        out, weights = self.token_attn(
            query=self.token_norm(tokens),
            key=self.context_norm(ctx),
            value=self.context_norm(ctx),
            need_weights=True,
            average_attn_weights=False,
        )
        pooled = out.mean(dim=1)
        summary = {
            "task_tokens/token_norm": float(tokens.norm(dim=-1).mean().item()),
            "task_tokens/context_norm": float(ctx.norm(dim=-1).mean().item()),
            "task_tokens/attention_entropy": float(_entropy_from_probs(weights.mean(dim=1), dim=-1).mean().item()),
        }
        return pooled, summary, {"task_tokens_attention": weights.mean(dim=1).detach()}


class _ResidualAttentionContextFusionBase(_AttentionContextFusionBase):
    """Shared residual-safe D0+FPN fusion base with a learnable residual scale."""

    attention_type = "residual_attention_context_base"

    def __init__(
        self,
        *,
        source_channels: dict[str, int],
        source_keys: Sequence[str],
        decoder_dim: int = 128,
        projection_dim: int = 64,
        alpha_init: float = 0.05,
    ) -> None:
        super().__init__(
            source_channels=source_channels,
            source_keys=source_keys,
            decoder_dim=decoder_dim,
            projection_dim=projection_dim,
        )
        alpha_init = float(min(max(alpha_init, 1e-4), 1.0 - 1e-4))
        self.raw_alpha = nn.Parameter(torch.tensor(math.log(alpha_init / (1.0 - alpha_init)), dtype=torch.float32))

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_alpha)

    def _combine(self, baseline: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return baseline + self.alpha * delta

    def forward(self, pyramid: dict[str, torch.Tensor]) -> torch.Tensor:
        projected, target_hw = self._project_sources(pyramid)
        baseline = self._baseline(projected)
        delta, summary, maps = self._context_delta(projected, baseline, target_hw)
        self._last_context_summary = {"residual/alpha": float(self.alpha.item()), **summary}
        self._last_attention_maps = maps
        fused = self._combine(baseline, delta)
        return self.head(self.body(fused))


class TaskTokensFiLMSpatialAfterHead(_ResidualAttentionContextFusionBase):
    attention_type = "task_tokens_film_then_spatial_attention"

    def __init__(
        self,
        *,
        source_channels: dict[str, int],
        source_keys: Sequence[str],
        decoder_dim: int = 128,
        projection_dim: int = 64,
        num_tokens: int = 4,
        num_heads: int = 4,
        pool_hw: int = 16,
        alpha_init: float = 0.05,
    ) -> None:
        super().__init__(
            source_channels=source_channels,
            source_keys=source_keys,
            decoder_dim=decoder_dim,
            projection_dim=projection_dim,
            alpha_init=alpha_init,
        )
        self.task_context = _TaskTokenContextModule(
            self.decoder_dim, num_tokens=num_tokens, num_heads=num_heads, pool_hw=pool_hw
        )
        self.gamma = nn.Linear(self.decoder_dim, self.decoder_dim)
        self.beta = nn.Linear(self.decoder_dim, self.decoder_dim)
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)
        _zero_init_conv(self.spatial)

    def _context_delta(self, projected, baseline, target_hw):
        task_vec, summary, maps = self.task_context(baseline)
        gamma = 1.0 + 0.1 * torch.tanh(self.gamma(task_vec)).view(-1, self.decoder_dim, 1, 1)
        beta = 0.1 * torch.tanh(self.beta(task_vec)).view(-1, self.decoder_dim, 1, 1)
        conditioned = baseline * gamma + beta
        avg = conditioned.mean(dim=1, keepdim=True)
        mx = conditioned.amax(dim=1, keepdim=True)
        spatial = 1.0 + 0.1 * torch.tanh(self.spatial(torch.cat([avg, mx], dim=1)))
        delta = conditioned * spatial - baseline
        summary.update({
            "task_tokens/gamma_mean": float(gamma.mean().item()),
            "task_tokens/beta_mean": float(beta.mean().item()),
            "spatial_attention/mean": float(spatial.mean().item()),
            "spatial_attention/std": float(spatial.std(unbiased=False).item()),
        })
        maps.update({"spatial_attention": spatial.detach(), "conditioned_features": conditioned.detach()})
        return delta, summary, maps


class SpatialBeforeTaskTokensFiLMHead(_ResidualAttentionContextFusionBase):
    attention_type = "spatial_attention_then_task_tokens_film"

    def __init__(
        self,
        *,
        source_channels: dict[str, int],
        source_keys: Sequence[str],
        decoder_dim: int = 128,
        projection_dim: int = 64,
        num_tokens: int = 4,
        num_heads: int = 4,
        pool_hw: int = 16,
        alpha_init: float = 0.05,
    ) -> None:
        super().__init__(
            source_channels=source_channels,
            source_keys=source_keys,
            decoder_dim=decoder_dim,
            projection_dim=projection_dim,
            alpha_init=alpha_init,
        )
        self.task_context = _TaskTokenContextModule(
            self.decoder_dim, num_tokens=num_tokens, num_heads=num_heads, pool_hw=pool_hw
        )
        self.gamma = nn.Linear(self.decoder_dim, self.decoder_dim)
        self.beta = nn.Linear(self.decoder_dim, self.decoder_dim)
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)
        _zero_init_conv(self.spatial)

    def _context_delta(self, projected, baseline, target_hw):
        avg = baseline.mean(dim=1, keepdim=True)
        mx = baseline.amax(dim=1, keepdim=True)
        spatial = 1.0 + 0.1 * torch.tanh(self.spatial(torch.cat([avg, mx], dim=1)))
        spatial_applied = baseline * spatial
        task_vec, summary, maps = self.task_context(spatial_applied)
        gamma = 1.0 + 0.1 * torch.tanh(self.gamma(task_vec)).view(-1, self.decoder_dim, 1, 1)
        beta = 0.1 * torch.tanh(self.beta(task_vec)).view(-1, self.decoder_dim, 1, 1)
        conditioned = spatial_applied * gamma + beta
        delta = conditioned - baseline
        summary.update({
            "task_tokens/gamma_mean": float(gamma.mean().item()),
            "task_tokens/beta_mean": float(beta.mean().item()),
            "spatial_attention/mean": float(spatial.mean().item()),
            "spatial_attention/std": float(spatial.std(unbiased=False).item()),
        })
        maps.update({"spatial_attention": spatial.detach(), "conditioned_features": conditioned.detach()})
        return delta, summary, maps


class TaskConditionedSpatialAttentionHead(_ResidualAttentionContextFusionBase):
    attention_type = "task_conditioned_spatial_attention"

    def __init__(
        self,
        *,
        source_channels: dict[str, int],
        source_keys: Sequence[str],
        decoder_dim: int = 128,
        projection_dim: int = 64,
        num_tokens: int = 4,
        num_heads: int = 4,
        pool_hw: int = 16,
        alpha_init: float = 0.05,
    ) -> None:
        super().__init__(
            source_channels=source_channels,
            source_keys=source_keys,
            decoder_dim=decoder_dim,
            projection_dim=projection_dim,
            alpha_init=alpha_init,
        )
        self.task_context = _TaskTokenContextModule(
            self.decoder_dim, num_tokens=num_tokens, num_heads=num_heads, pool_hw=pool_hw
        )
        self.context_channel = nn.Linear(self.decoder_dim, self.decoder_dim)
        self.context_spatial = nn.Linear(self.decoder_dim, 1)
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        nn.init.zeros_(self.context_channel.weight)
        nn.init.zeros_(self.context_channel.bias)
        nn.init.zeros_(self.context_spatial.weight)
        nn.init.zeros_(self.context_spatial.bias)
        _zero_init_conv(self.spatial)

    def _context_delta(self, projected, baseline, target_hw):
        task_vec, summary, maps = self.task_context(baseline)
        channel_gate = 1.0 + 0.1 * torch.tanh(self.context_channel(task_vec)).view(-1, self.decoder_dim, 1, 1)
        task_bias = self.context_spatial(task_vec).view(-1, 1, 1, 1)
        avg = (baseline * channel_gate).mean(dim=1, keepdim=True)
        mx = (baseline * channel_gate).amax(dim=1, keepdim=True)
        spatial = 1.0 + 0.1 * torch.tanh(self.spatial(torch.cat([avg, mx], dim=1)) + task_bias)
        conditioned = baseline * channel_gate * spatial
        delta = conditioned - baseline
        summary.update({
            "task_tokens/channel_mean": float(channel_gate.mean().item()),
            "task_tokens/context_bias": float(task_bias.mean().item()),
            "spatial_attention/mean": float(spatial.mean().item()),
            "spatial_attention/std": float(spatial.std(unbiased=False).item()),
        })
        maps.update({
            "task_conditioned_spatial_attention": spatial.detach(),
            "context_channel_gate": channel_gate.detach(),
        })
        return delta, summary, maps


class ResidualTaskSpatialDeltaHead(_ResidualAttentionContextFusionBase):
    attention_type = "residual_task_spatial_delta"

    def __init__(
        self,
        *,
        source_channels: dict[str, int],
        source_keys: Sequence[str],
        decoder_dim: int = 128,
        projection_dim: int = 64,
        num_tokens: int = 4,
        num_heads: int = 4,
        pool_hw: int = 16,
        alpha_init: float = 0.05,
    ) -> None:
        super().__init__(
            source_channels=source_channels,
            source_keys=source_keys,
            decoder_dim=decoder_dim,
            projection_dim=projection_dim,
            alpha_init=alpha_init,
        )
        self.task_context = _TaskTokenContextModule(
            self.decoder_dim, num_tokens=num_tokens, num_heads=num_heads, pool_hw=pool_hw
        )
        self.gamma = nn.Linear(self.decoder_dim, self.decoder_dim)
        self.beta = nn.Linear(self.decoder_dim, self.decoder_dim)
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.delta_refine = nn.Sequential(
            ConvBNReLU(self.decoder_dim, self.decoder_dim),
            ConvBNReLU(self.decoder_dim, self.decoder_dim),
            nn.Conv2d(self.decoder_dim, self.decoder_dim, kernel_size=1),
        )
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)
        _zero_init_conv(self.spatial)
        _zero_init_conv(self.delta_refine[-1])

    def _context_delta(self, projected, baseline, target_hw):
        task_vec, summary, maps = self.task_context(baseline)
        gamma = 1.0 + 0.1 * torch.tanh(self.gamma(task_vec)).view(-1, self.decoder_dim, 1, 1)
        beta = 0.1 * torch.tanh(self.beta(task_vec)).view(-1, self.decoder_dim, 1, 1)
        conditioned = baseline * gamma + beta
        avg = conditioned.mean(dim=1, keepdim=True)
        mx = conditioned.amax(dim=1, keepdim=True)
        spatial = 1.0 + 0.1 * torch.tanh(self.spatial(torch.cat([avg, mx], dim=1)))
        refined = conditioned * spatial
        delta = self.delta_refine(refined - baseline)
        summary.update({
            "task_tokens/gamma_mean": float(gamma.mean().item()),
            "task_tokens/beta_mean": float(beta.mean().item()),
            "spatial_attention/mean": float(spatial.mean().item()),
            "spatial_attention/std": float(spatial.std(unbiased=False).item()),
        })
        maps.update({"spatial_attention": spatial.detach(), "residual_delta": delta.detach()})
        return delta, summary, maps


class LogitResidualTaskSpatialHead(_AttentionContextFusionBase):
    attention_type = "logit_residual_task_spatial_correction"

    def __init__(
        self,
        *,
        source_channels: dict[str, int],
        source_keys: Sequence[str],
        decoder_dim: int = 128,
        projection_dim: int = 64,
        num_tokens: int = 4,
        num_heads: int = 4,
        pool_hw: int = 16,
        alpha_init: float = 0.05,
    ) -> None:
        super().__init__(
            source_channels=source_channels,
            source_keys=source_keys,
            decoder_dim=decoder_dim,
            projection_dim=projection_dim,
        )
        alpha_init = float(min(max(alpha_init, 1e-4), 1.0 - 1e-4))
        self.raw_alpha = nn.Parameter(torch.tensor(math.log(alpha_init / (1.0 - alpha_init)), dtype=torch.float32))
        self.base_head = DecoderSemanticFpnFusionHead(
            f2_channels=int(source_channels["fpn_2"]),
            f1_channels=int(source_channels["fpn_1"]),
            f0_channels=int(source_channels["fpn_0"]),
            decoder_map_channels=int(source_channels["decoder_semantic_map"]),
            decoder_dim=decoder_dim,
            projection_dim=projection_dim,
        )
        self.task_context = _TaskTokenContextModule(
            self.decoder_dim, num_tokens=num_tokens, num_heads=num_heads, pool_hw=pool_hw
        )
        self.gamma = nn.Linear(self.decoder_dim, self.decoder_dim)
        self.beta = nn.Linear(self.decoder_dim, self.decoder_dim)
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.delta_head = nn.Sequential(
            ConvBNReLU(self.decoder_dim, self.decoder_dim),
            ConvBNReLU(self.decoder_dim, self.decoder_dim),
            nn.Conv2d(self.decoder_dim, 1, kernel_size=1),
        )
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)
        _zero_init_conv(self.spatial)
        _zero_init_conv(self.delta_head[-1])

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_alpha)

    def forward(self, pyramid: dict[str, torch.Tensor]) -> torch.Tensor:
        projected, target_hw = self._project_sources(pyramid)
        baseline = self._baseline(projected)
        task_vec, summary, maps = self.task_context(baseline)
        gamma = 1.0 + 0.1 * torch.tanh(self.gamma(task_vec)).view(-1, self.decoder_dim, 1, 1)
        beta = 0.1 * torch.tanh(self.beta(task_vec)).view(-1, self.decoder_dim, 1, 1)
        conditioned = baseline * gamma + beta
        avg = conditioned.mean(dim=1, keepdim=True)
        mx = conditioned.amax(dim=1, keepdim=True)
        spatial = 1.0 + 0.1 * torch.tanh(self.spatial(torch.cat([avg, mx], dim=1)))
        refined = conditioned * spatial
        base_logits = self.base_head(pyramid)
        delta_logits = self.delta_head(refined - baseline)
        self._last_context_summary = {
            "residual/alpha": float(self.alpha.item()),
            "task_tokens/gamma_mean": float(gamma.mean().item()),
            "task_tokens/beta_mean": float(beta.mean().item()),
            "spatial_attention/mean": float(spatial.mean().item()),
            "spatial_attention/std": float(spatial.std(unbiased=False).item()),
            **summary,
        }
        self._last_attention_maps = {
            **maps,
            "spatial_attention": spatial.detach(),
            "delta_logits": delta_logits.detach(),
        }
        return base_logits + self.alpha * delta_logits


class DualSkipTaskSpatialFusionHead(_ResidualAttentionContextFusionBase):
    attention_type = "dual_skip_task_spatial_fusion"

    def __init__(
        self,
        *,
        source_channels: dict[str, int],
        source_keys: Sequence[str],
        decoder_dim: int = 128,
        projection_dim: int = 64,
        num_tokens: int = 4,
        num_heads: int = 4,
        pool_hw: int = 16,
        alpha_init: float = 0.05,
    ) -> None:
        super().__init__(
            source_channels=source_channels,
            source_keys=source_keys,
            decoder_dim=decoder_dim,
            projection_dim=projection_dim,
            alpha_init=alpha_init,
        )
        self.task_context = _TaskTokenContextModule(
            self.decoder_dim, num_tokens=num_tokens, num_heads=num_heads, pool_hw=pool_hw
        )
        self.gamma = nn.Linear(self.decoder_dim, self.decoder_dim)
        self.beta = nn.Linear(self.decoder_dim, self.decoder_dim)
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.dual_fuse = nn.Sequential(
            ConvBNReLU(self.decoder_dim * 4, self.decoder_dim),
            ConvBNReLU(self.decoder_dim, self.decoder_dim),
            nn.Conv2d(self.decoder_dim, self.decoder_dim, kernel_size=1),
        )
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)
        _zero_init_conv(self.spatial)
        _zero_init_conv(self.dual_fuse[-1])

    def _context_delta(self, projected, baseline, target_hw):
        task_vec, summary, maps = self.task_context(baseline)
        gamma = 1.0 + 0.1 * torch.tanh(self.gamma(task_vec)).view(-1, self.decoder_dim, 1, 1)
        beta = 0.1 * torch.tanh(self.beta(task_vec)).view(-1, self.decoder_dim, 1, 1)
        conditioned = baseline * gamma + beta
        avg = conditioned.mean(dim=1, keepdim=True)
        mx = conditioned.amax(dim=1, keepdim=True)
        spatial = 1.0 + 0.1 * torch.tanh(self.spatial(torch.cat([avg, mx], dim=1)))
        refined = conditioned * spatial
        combined = torch.cat([baseline, refined, baseline - refined, baseline * refined], dim=1)
        delta = self.dual_fuse(combined)
        summary.update({
            "task_tokens/gamma_mean": float(gamma.mean().item()),
            "task_tokens/beta_mean": float(beta.mean().item()),
            "spatial_attention/mean": float(spatial.mean().item()),
            "spatial_attention/std": float(spatial.std(unbiased=False).item()),
        })
        maps.update({"spatial_attention": spatial.detach(), "dual_skip_delta": delta.detach()})
        return delta, summary, maps


class SourceSpatialGatesTaskTokensHead(_ResidualAttentionContextFusionBase):
    attention_type = "source_spatial_gates_task_tokens"

    def __init__(
        self,
        *,
        source_channels: dict[str, int],
        source_keys: Sequence[str],
        decoder_dim: int = 128,
        projection_dim: int = 64,
        num_tokens: int = 4,
        num_heads: int = 4,
        pool_hw: int = 16,
        alpha_init: float = 0.05,
    ) -> None:
        super().__init__(
            source_channels=source_channels,
            source_keys=source_keys,
            decoder_dim=decoder_dim,
            projection_dim=projection_dim,
            alpha_init=alpha_init,
        )
        self.task_context = _TaskTokenContextModule(
            self.decoder_dim, num_tokens=num_tokens, num_heads=num_heads, pool_hw=pool_hw
        )
        self.source_channel_gate = nn.ModuleDict(
            {key: nn.Linear(self.decoder_dim, self.projection_dim) for key in self.source_keys}
        )
        self.source_spatial_gate = nn.ModuleDict(
            {key: nn.Conv2d(2, 1, kernel_size=7, padding=3) for key in self.source_keys}
        )
        for gate in self.source_channel_gate.values():
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)
        for gate in self.source_spatial_gate.values():
            _zero_init_conv(gate)

    def _context_delta(self, projected, baseline, target_hw):
        task_vec, summary, maps = self.task_context(baseline)
        gated_parts = []
        for key in self.source_keys:
            x = projected[key]
            channel_gate = 1.0 + 0.1 * torch.tanh(self.source_channel_gate[key](task_vec)).view(-1, self.projection_dim, 1, 1)
            avg = x.mean(dim=1, keepdim=True)
            mx = x.amax(dim=1, keepdim=True)
            spatial = 1.0 + 0.1 * torch.tanh(self.source_spatial_gate[key](torch.cat([avg, mx], dim=1)))
            gated_parts.append(x * channel_gate * spatial)
            summary[f"source_gate/{key}_channel_mean"] = float(channel_gate.mean().item())
            summary[f"source_gate/{key}_spatial_mean"] = float(spatial.mean().item())
            maps[f"source_gate_{key}"] = (channel_gate.mean(dim=1, keepdim=True) * spatial).detach()
        gated = torch.cat(gated_parts, dim=1)
        delta = self.pre_fuse(gated) - baseline
        return delta, summary, maps


class F0BoundarySpatialTaskRefineHead(_ResidualAttentionContextFusionBase):
    attention_type = "f0_boundary_spatial_task_refine"

    def __init__(
        self,
        *,
        source_channels: dict[str, int],
        source_keys: Sequence[str],
        decoder_dim: int = 128,
        projection_dim: int = 64,
        num_tokens: int = 4,
        num_heads: int = 4,
        pool_hw: int = 16,
        alpha_init: float = 0.05,
    ) -> None:
        super().__init__(
            source_channels=source_channels,
            source_keys=source_keys,
            decoder_dim=decoder_dim,
            projection_dim=projection_dim,
            alpha_init=alpha_init,
        )
        if "fpn_0" not in self.source_keys:
            raise ValueError("F0BoundarySpatialTaskRefineHead requires fpn_0 in source_keys")
        self.task_context = _TaskTokenContextModule(
            self.decoder_dim, num_tokens=num_tokens, num_heads=num_heads, pool_hw=pool_hw
        )
        self.f0_channel_gate = nn.Linear(self.decoder_dim, self.projection_dim)
        self.f0_spatial_gate = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.f0_refine = nn.Sequential(
            ConvBNReLU(self.projection_dim, self.projection_dim),
            ConvBNReLU(self.projection_dim, self.projection_dim),
            nn.Conv2d(self.projection_dim, self.projection_dim, kernel_size=1),
        )
        nn.init.zeros_(self.f0_channel_gate.weight)
        nn.init.zeros_(self.f0_channel_gate.bias)
        _zero_init_conv(self.f0_spatial_gate)
        _zero_init_conv(self.f0_refine[-1])

    def _context_delta(self, projected, baseline, target_hw):
        task_vec, summary, maps = self.task_context(baseline)
        f0 = projected["fpn_0"]
        channel_gate = 1.0 + 0.1 * torch.tanh(self.f0_channel_gate(task_vec)).view(-1, self.projection_dim, 1, 1)
        avg = f0.mean(dim=1, keepdim=True)
        mx = f0.amax(dim=1, keepdim=True)
        spatial = 1.0 + 0.1 * torch.tanh(self.f0_spatial_gate(torch.cat([avg, mx], dim=1)))
        refined_f0 = f0 * channel_gate * spatial
        gated = dict(projected)
        gated["fpn_0"] = refined_f0
        delta = self.pre_fuse(torch.cat([gated[key] for key in self.source_keys], dim=1)) - baseline
        delta = self.f0_refine(delta)
        summary.update({
            "f0_gate/channel_mean": float(channel_gate.mean().item()),
            "f0_gate/spatial_mean": float(spatial.mean().item()),
            "f0_gate/spatial_std": float(spatial.std(unbiased=False).item()),
        })
        maps.update({"f0_boundary_spatial_attention": spatial.detach(), "f0_refined": refined_f0.detach()})
        return delta, summary, maps


AttentionContextFusionHeadBase = _AttentionContextFusionBase

__all__ = [
    "AttentionContextFusionHeadBase",
    "BidirectionalDecoderEncoderCrossAttentionHead",
    "CbamSpatialChannelAttentionFusionHead",
    "D0QueryFpnCrossAttentionHead",
    "FpnQueryD0CrossAttentionHead",
    "LearnedTaskTokensFiLMHead",
    "TaskTokensFiLMSpatialAfterHead",
    "SpatialBeforeTaskTokensFiLMHead",
    "TaskConditionedSpatialAttentionHead",
    "ResidualTaskSpatialDeltaHead",
    "LogitResidualTaskSpatialHead",
    "DualSkipTaskSpatialFusionHead",
    "SourceSpatialGatesTaskTokensHead",
    "F0BoundarySpatialTaskRefineHead",
    "LowRankGlobalContextFusionHead",
    "SourceGatedChannelContextHead",
    "WindowSelfAttentionFusionHead",
]
