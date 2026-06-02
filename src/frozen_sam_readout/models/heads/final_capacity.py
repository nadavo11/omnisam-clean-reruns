"""Final MoNuSeg capacity-sweep heads.

These variants extend the frozen D0+FPN fusion family with a few higher-capacity
or deeper residual readouts while preserving a safe baseline path.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..blocks.conv_blocks import ConvBNReLU
from .attention_context import (
    _AttentionContextFusionBase,
    _TaskTokenContextModule,
    _resize_to,
    _zero_init_conv,
)
from .decoder_semantic_map import DecoderSemanticFpnFusionHead


def _schedule_value(epoch: int, warmup_epochs: int, *, device: torch.device) -> torch.Tensor:
    if warmup_epochs <= 0:
        return torch.tensor(1.0, device=device)
    value = min(max(float(epoch) / float(warmup_epochs), 0.0), 1.0)
    return torch.tensor(value, device=device)


class _ScheduledResidualAttentionContextFusionBase(_AttentionContextFusionBase):
    """Residual-safe fusion base with an explicit epoch-based alpha schedule."""

    attention_type = "scheduled_residual_attention_context_base"

    def __init__(
        self,
        *,
        source_channels: dict[str, int],
        source_keys: Sequence[str],
        decoder_dim: int = 128,
        projection_dim: int = 64,
        warmup_epochs: int = 20,
    ) -> None:
        super().__init__(
            source_channels=source_channels,
            source_keys=source_keys,
            decoder_dim=decoder_dim,
            projection_dim=projection_dim,
        )
        self.warmup_epochs = int(warmup_epochs)
        self._current_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._current_epoch = int(epoch)

    @property
    def alpha(self) -> torch.Tensor:
        device = next(self.parameters()).device
        return _schedule_value(self._current_epoch, self.warmup_epochs, device=device)

    def forward(self, pyramid: dict[str, torch.Tensor]) -> torch.Tensor:
        projected, target_hw = self._project_sources(pyramid)
        baseline = self._baseline(projected)
        delta, summary, maps = self._context_delta(projected, baseline, target_hw)
        self._last_context_summary = {"residual/alpha": float(self.alpha.item()), **summary}
        self._last_attention_maps = maps
        fused = baseline + self.alpha * delta
        return self.head(self.body(fused))


class _ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(channels, channels),
            ConvBNReLU(channels, channels),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        _zero_init_conv(self.block[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class _TaskSpatialBlock(nn.Module):
    def __init__(
        self,
        *,
        decoder_dim: int,
        num_tokens: int,
        num_heads: int,
        pool_hw: int,
        refine_depth: int = 2,
        scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.decoder_dim = int(decoder_dim)
        self.scale = float(scale)
        self.task_context = _TaskTokenContextModule(
            self.decoder_dim, num_tokens=num_tokens, num_heads=num_heads, pool_hw=pool_hw
        )
        self.gamma = nn.Linear(self.decoder_dim, self.decoder_dim)
        self.beta = nn.Linear(self.decoder_dim, self.decoder_dim)
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        refine_layers = []
        for _ in range(max(1, int(refine_depth))):
            refine_layers.append(ConvBNReLU(self.decoder_dim, self.decoder_dim))
        refine_layers.append(nn.Conv2d(self.decoder_dim, self.decoder_dim, kernel_size=1))
        self.refine = nn.Sequential(*refine_layers)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)
        _zero_init_conv(self.spatial)
        _zero_init_conv(self.refine[-1])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
        task_vec, summary, maps = self.task_context(x)
        gamma = 1.0 + self.scale * torch.tanh(self.gamma(task_vec)).view(-1, self.decoder_dim, 1, 1)
        beta = self.scale * torch.tanh(self.beta(task_vec)).view(-1, self.decoder_dim, 1, 1)
        conditioned = x * gamma + beta
        avg = conditioned.mean(dim=1, keepdim=True)
        mx = conditioned.amax(dim=1, keepdim=True)
        spatial = 1.0 + self.scale * torch.tanh(self.spatial(torch.cat([avg, mx], dim=1)))
        refined = conditioned * spatial
        delta = self.refine(refined - x)
        summary.update(
            {
                "task_tokens/gamma_mean": float(gamma.mean().item()),
                "task_tokens/beta_mean": float(beta.mean().item()),
                "task_tokens/token_norm": float(task_vec.norm(dim=-1).mean().item()),
                "task_tokens/attention_entropy": float(summary.get("task_tokens/attention_entropy", float("nan"))),
                "spatial_attention/mean": float(spatial.mean().item()),
                "spatial_attention/std": float(spatial.std(unbiased=False).item()),
            }
        )
        maps.update({"spatial_attention": spatial.detach(), "task_conditioned_features": refined.detach()})
        return delta, summary, maps


class _F0BoundaryRefineBlock(nn.Module):
    def __init__(
        self,
        *,
        decoder_dim: int,
        projection_dim: int,
        num_tokens: int,
        num_heads: int,
        pool_hw: int,
        scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.decoder_dim = int(decoder_dim)
        self.projection_dim = int(projection_dim)
        self.scale = float(scale)
        self.task_context = _TaskTokenContextModule(
            self.decoder_dim, num_tokens=num_tokens, num_heads=num_heads, pool_hw=pool_hw
        )
        self.f0_channel_gate = nn.Linear(self.decoder_dim, self.projection_dim)
        self.f0_spatial_gate = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.refine = nn.Sequential(
            ConvBNReLU(self.decoder_dim + self.projection_dim, self.decoder_dim),
            ConvBNReLU(self.decoder_dim, self.decoder_dim),
            nn.Conv2d(self.decoder_dim, self.decoder_dim, kernel_size=1),
        )
        nn.init.zeros_(self.f0_channel_gate.weight)
        nn.init.zeros_(self.f0_channel_gate.bias)
        _zero_init_conv(self.f0_spatial_gate)
        _zero_init_conv(self.refine[-1])

    def forward(
        self, baseline: torch.Tensor, projected: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
        task_vec, summary, maps = self.task_context(baseline)
        f0 = projected["fpn_0"]
        channel_gate = 1.0 + self.scale * torch.tanh(self.f0_channel_gate(task_vec)).view(-1, self.projection_dim, 1, 1)
        avg = f0.mean(dim=1, keepdim=True)
        mx = f0.amax(dim=1, keepdim=True)
        spatial = 1.0 + self.scale * torch.tanh(self.f0_spatial_gate(torch.cat([avg, mx], dim=1)))
        refined_f0 = f0 * channel_gate * spatial
        refined_f0 = _resize_to(refined_f0, baseline.shape[-2:])
        delta = self.refine(torch.cat([baseline, refined_f0], dim=1))
        summary.update(
            {
                "f0_gate/channel_mean": float(channel_gate.mean().item()),
                "f0_gate/spatial_mean": float(spatial.mean().item()),
                "f0_gate/spatial_std": float(spatial.std(unbiased=False).item()),
            }
        )
        maps.update({"f0_boundary_spatial_attention": spatial.detach(), "f0_refined": refined_f0.detach()})
        return delta, summary, maps


class TaskTokensFiLMWideHead(_ScheduledResidualAttentionContextFusionBase):
    attention_type = "cap160_task_tokens_film_wide"

    def __init__(
        self,
        *,
        source_channels: dict[str, int],
        source_keys: Sequence[str],
        decoder_dim: int = 192,
        projection_dim: int = 96,
        num_tokens: int = 16,
        num_heads: int = 8,
        pool_hw: int = 16,
        token_hidden_dim: int = 512,
        warmup_epochs: int = 20,
    ) -> None:
        super().__init__(
            source_channels=source_channels,
            source_keys=source_keys,
            decoder_dim=decoder_dim,
            projection_dim=projection_dim,
            warmup_epochs=warmup_epochs,
        )
        self.token_hidden_dim = int(token_hidden_dim)
        self.task_context = _TaskTokenContextModule(
            self.decoder_dim, num_tokens=num_tokens, num_heads=num_heads, pool_hw=pool_hw
        )
        self.film_mlp = nn.Sequential(
            nn.Linear(self.decoder_dim, self.token_hidden_dim),
            nn.GELU(),
            nn.Linear(self.token_hidden_dim, self.token_hidden_dim),
            nn.GELU(),
        )
        self.gamma = nn.Linear(self.token_hidden_dim, self.decoder_dim)
        self.beta = nn.Linear(self.token_hidden_dim, self.decoder_dim)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def _context_delta(self, projected, baseline, target_hw):
        task_vec, summary, maps = self.task_context(baseline)
        hidden = self.film_mlp(task_vec)
        gamma = 1.0 + 0.1 * torch.tanh(self.gamma(hidden)).view(-1, self.decoder_dim, 1, 1)
        beta = 0.1 * torch.tanh(self.beta(hidden)).view(-1, self.decoder_dim, 1, 1)
        delta = baseline * gamma + beta - baseline
        summary.update(
            {
                "task_tokens/hidden_norm": float(hidden.norm(dim=-1).mean().item()),
                "task_tokens/gamma_mean": float(gamma.mean().item()),
                "task_tokens/beta_mean": float(beta.mean().item()),
                "task_tokens/token_hidden_dim": float(self.token_hidden_dim),
            }
        )
        return delta, summary, maps


class TaskSpatialResidualStackHead(_ScheduledResidualAttentionContextFusionBase):
    attention_type = "cap160_task_spatial_residual_stack"

    def __init__(
        self,
        *,
        source_channels: dict[str, int],
        source_keys: Sequence[str],
        decoder_dim: int = 192,
        projection_dim: int = 96,
        num_tokens: int = 8,
        num_heads: int = 8,
        pool_hw: int = 16,
        num_blocks: int = 2,
        refine_depth: int = 2,
        warmup_epochs: int = 20,
    ) -> None:
        super().__init__(
            source_channels=source_channels,
            source_keys=source_keys,
            decoder_dim=decoder_dim,
            projection_dim=projection_dim,
            warmup_epochs=warmup_epochs,
        )
        self.blocks = nn.ModuleList(
            [
                _TaskSpatialBlock(
                    decoder_dim=self.decoder_dim,
                    num_tokens=num_tokens,
                    num_heads=num_heads,
                    pool_hw=pool_hw,
                    refine_depth=refine_depth,
                )
                for _ in range(int(num_blocks))
            ]
        )

    def _context_delta(self, projected, baseline, target_hw):
        x = baseline
        summary: dict[str, float] = {}
        maps: dict[str, torch.Tensor] = {}
        for idx, block in enumerate(self.blocks):
            delta, block_summary, block_maps = block(x)
            x = x + delta
            summary.update({f"block{idx}/{k}": v for k, v in block_summary.items()})
            for key, value in block_maps.items():
                maps[f"block{idx}/{key}"] = value
        return x - baseline, summary, maps


class FineMapFusionHighResHead(_ScheduledResidualAttentionContextFusionBase):
    attention_type = "cap160_fine_map_fusion_highres"

    def __init__(
        self,
        *,
        source_channels: dict[str, int],
        source_keys: Sequence[str],
        decoder_dim: int = 192,
        projection_dim: int = 96,
        warmup_epochs: int = 20,
    ) -> None:
        super().__init__(
            source_channels=source_channels,
            source_keys=source_keys,
            decoder_dim=decoder_dim,
            projection_dim=projection_dim,
            warmup_epochs=warmup_epochs,
        )
        if "fpn_0" not in self.source_keys:
            raise ValueError("FineMapFusionHighResHead requires fpn_0 in source_keys")
        self.highres_body = nn.Sequential(
            ConvBNReLU(self.projection_dim * len(self.source_keys), self.decoder_dim),
            ConvBNReLU(self.decoder_dim, self.decoder_dim),
            ConvBNReLU(self.decoder_dim, self.decoder_dim),
        )
        self.highres_head = nn.Conv2d(self.decoder_dim, 1, kernel_size=1)
        _zero_init_conv(self.highres_head)

    def forward(self, pyramid: dict[str, torch.Tensor]) -> torch.Tensor:
        projected, target_hw = self._project_sources(pyramid)
        lowres_base = self._baseline(projected)
        highres_hw = projected["fpn_0"].shape[-2:]
        parts = []
        for key in self.source_keys:
            x = projected[key]
            if x.shape[-2:] != highres_hw:
                x = _resize_to(x, highres_hw)
            parts.append(x)
        highres_base = _resize_to(lowres_base, highres_hw)
        highres_fused = self.highres_body(torch.cat(parts, dim=1))
        delta = highres_fused - highres_base
        self._last_context_summary = {
            "residual/alpha": float(self.alpha.item()),
            "highres/base_hw_h": float(highres_hw[0]),
            "highres/base_hw_w": float(highres_hw[1]),
            "highres/fusion_width": float(self.decoder_dim),
        }
        self._last_attention_maps = {"highres_features": highres_fused.detach()}
        fused = highres_base + self.alpha * delta
        return self.highres_head(fused)


class DeepResidualConvFusionHead(_ScheduledResidualAttentionContextFusionBase):
    attention_type = "cap160_deep_conv_fusion_residual"

    def __init__(
        self,
        *,
        source_channels: dict[str, int],
        source_keys: Sequence[str],
        decoder_dim: int = 192,
        projection_dim: int = 96,
        num_blocks: int = 8,
        task_every: int = 2,
        num_tokens: int = 8,
        num_heads: int = 8,
        pool_hw: int = 16,
        warmup_epochs: int = 20,
    ) -> None:
        super().__init__(
            source_channels=source_channels,
            source_keys=source_keys,
            decoder_dim=decoder_dim,
            projection_dim=projection_dim,
            warmup_epochs=warmup_epochs,
        )
        self.num_blocks = int(num_blocks)
        self.task_every = int(task_every)
        self.task_context = _TaskTokenContextModule(
            self.decoder_dim, num_tokens=num_tokens, num_heads=num_heads, pool_hw=pool_hw
        )
        self.gamma = nn.Linear(self.decoder_dim, self.decoder_dim)
        self.beta = nn.Linear(self.decoder_dim, self.decoder_dim)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)
        self.blocks = nn.ModuleList([_ResidualConvBlock(self.decoder_dim) for _ in range(self.num_blocks)])

    def _context_delta(self, projected, baseline, target_hw):
        x = baseline
        summary: dict[str, float] = {}
        maps: dict[str, torch.Tensor] = {}
        for idx, block in enumerate(self.blocks):
            if self.task_every > 0 and idx % self.task_every == 0:
                task_vec, task_summary, task_maps = self.task_context(x)
                gamma = 1.0 + 0.1 * torch.tanh(self.gamma(task_vec)).view(-1, self.decoder_dim, 1, 1)
                beta = 0.1 * torch.tanh(self.beta(task_vec)).view(-1, self.decoder_dim, 1, 1)
                x = x * gamma + beta
                summary.update({
                    f"block{idx}/task_tokens/gamma_mean": float(gamma.mean().item()),
                    f"block{idx}/task_tokens/beta_mean": float(beta.mean().item()),
                    f"block{idx}/task_tokens/attention_entropy": float(task_summary["task_tokens/attention_entropy"]),
                })
                maps[f"block{idx}_task_tokens_attention"] = task_maps["task_tokens_attention"]
            x = block(x)
        return x - baseline, summary, maps


class HybridTaskSpatialFineHead(_ScheduledResidualAttentionContextFusionBase):
    attention_type = "cap160_hybrid_task_spatial_fine"

    def __init__(
        self,
        *,
        source_channels: dict[str, int],
        source_keys: Sequence[str],
        decoder_dim: int = 192,
        projection_dim: int = 96,
        num_tokens: int = 16,
        num_heads: int = 8,
        pool_hw: int = 16,
        task_blocks: int = 2,
        refine_depth: int = 2,
        warmup_epochs: int = 20,
    ) -> None:
        super().__init__(
            source_channels=source_channels,
            source_keys=source_keys,
            decoder_dim=decoder_dim,
            projection_dim=projection_dim,
            warmup_epochs=warmup_epochs,
        )
        self.task_blocks = nn.ModuleList(
            [
                _TaskSpatialBlock(
                    decoder_dim=self.decoder_dim,
                    num_tokens=num_tokens,
                    num_heads=num_heads,
                    pool_hw=pool_hw,
                    refine_depth=refine_depth,
                )
                for _ in range(int(task_blocks))
            ]
        )
        self.f0_branch = _F0BoundaryRefineBlock(
            decoder_dim=self.decoder_dim,
            projection_dim=self.projection_dim,
            num_tokens=num_tokens,
            num_heads=num_heads,
            pool_hw=pool_hw,
        )

    def _context_delta(self, projected, baseline, target_hw):
        x = baseline
        summary: dict[str, float] = {}
        maps: dict[str, torch.Tensor] = {}
        for idx, block in enumerate(self.task_blocks):
            delta, block_summary, block_maps = block(x)
            x = x + delta
            summary.update({f"task_block{idx}/{k}": v for k, v in block_summary.items()})
            for key, value in block_maps.items():
                maps[f"task_block{idx}_{key}"] = value
        f0_delta, f0_summary, f0_maps = self.f0_branch(x, projected)
        summary.update({f"f0/{k}": v for k, v in f0_summary.items()})
        maps.update({f"f0_{k}": v for k, v in f0_maps.items()})
        return (x - baseline) + f0_delta, summary, maps


class WideDecoderSemanticFpnFusionHead(DecoderSemanticMultiFusionHead):
    """Widened full D0+FPN baseline with the generic multi-fusion interface."""

    attention_type = "cap160_wide_fusion_dim256"

    def __init__(
        self,
        *,
        source_channels: dict[str, int],
        source_keys: tuple[str, ...],
        decoder_dim: int = 128,
        projection_dim: int = 256,
        **kwargs: int,
    ) -> None:
        del kwargs
        super().__init__(
            source_channels=source_channels,
            source_keys=source_keys,
            decoder_dim=decoder_dim,
            projection_dim=projection_dim,
        )


__all__ = [
    "TaskTokensFiLMWideHead",
    "TaskSpatialResidualStackHead",
    "FineMapFusionHighResHead",
    "DeepResidualConvFusionHead",
    "HybridTaskSpatialFineHead",
    "WideDecoderSemanticFpnFusionHead",
]
