"""Model registry mapping variant names to head classes."""

from __future__ import annotations

from typing import Any, Dict, Type

import torch.nn as nn

from .heads import (
    A0Fpn2Head,
    A2Fpn2Fpn1RefineHead,
    A3MemoryHead,
    A5F0ResidualHead,
    BidirectionalDecoderEncoderCrossAttentionHead,
    CbamSpatialChannelAttentionFusionHead,
    D0QueryFpnCrossAttentionHead,
    DecoderSemanticF0FusionHead,
    DecoderSemanticFpnFusionHead,
    DecoderSemanticMultiFusionHead,
    DecoderSemanticLinearHead,
    DecoderSemanticSmallHead,
    FpnPlusDecoderSemanticHead,
    FpnQueryD0CrossAttentionHead,
    LearnedTaskTokensFiLMHead,
    LowRankGlobalContextFusionHead,
    GatedAllRefineHead,
    GatedWarmupHead,
    SourceGatedChannelContextHead,
    StagedA2ReadoutHead,
    StagedMultiscaleReadoutHead,
    WindowSelfAttentionFusionHead,
)


MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "a0_fpn2": A0Fpn2Head,
    "a2_fpn2_fpn1_refine": A2Fpn2Fpn1RefineHead,
    "a3_memory": A3MemoryHead,
    "a5_all_at_once": A5F0ResidualHead,
    "final_staged": StagedMultiscaleReadoutHead,
    "final_staged_a2": StagedA2ReadoutHead,
    "gated_a3_f0_warmup": GatedWarmupHead,
    "gated_all_refine_mem": GatedAllRefineHead,
    "d0_linear": DecoderSemanticLinearHead,
    "d0_small_head": DecoderSemanticSmallHead,
    "fpn_plus_d0": FpnPlusDecoderSemanticHead,
    "d0_fpn_fusion": DecoderSemanticFpnFusionHead,
    "d0_f0_fusion": DecoderSemanticF0FusionHead,
    "d0_f1_fusion": DecoderSemanticMultiFusionHead,
    "d0_f2_fusion": DecoderSemanticMultiFusionHead,
    "d0_f0_f1_fusion": DecoderSemanticMultiFusionHead,
    "d0_f1_f2_fusion": DecoderSemanticMultiFusionHead,
    "d0_f0_f2_fusion": DecoderSemanticMultiFusionHead,
    "d0fpn_source_gated_se": SourceGatedChannelContextHead,
    "d0fpn_cbam": CbamSpatialChannelAttentionFusionHead,
    "d0_query_fpn_crossattn": D0QueryFpnCrossAttentionHead,
    "fpn_query_d0_crossattn": FpnQueryD0CrossAttentionHead,
    "d0fpn_bidirectional_crossattn": BidirectionalDecoderEncoderCrossAttentionHead,
    "d0fpn_task_tokens_film": LearnedTaskTokensFiLMHead,
    "d0fpn_window_selfattn": WindowSelfAttentionFusionHead,
    "d0fpn_global_context_nonlocal": LowRankGlobalContextFusionHead,
}


def build_head(variant: str, **kwargs: Any) -> nn.Module:
    """Construct a head by variant name with the given channel dims and config."""

    if variant not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown variant '{variant}'. Known: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[variant](**kwargs)
