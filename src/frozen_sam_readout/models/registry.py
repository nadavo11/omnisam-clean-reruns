"""Model registry mapping variant names to head classes."""

from __future__ import annotations

from typing import Any, Dict, Type

import torch.nn as nn

from .heads import (
    A0Fpn2Head,
    A2Fpn2Fpn1RefineHead,
    A3MemoryHead,
    A5F0ResidualHead,
    GatedAllRefineHead,
    GatedWarmupHead,
    StagedA2ReadoutHead,
    StagedMultiscaleReadoutHead,
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
}


def build_head(variant: str, **kwargs: Any) -> nn.Module:
    """Construct a head by variant name with the given channel dims and config."""

    if variant not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown variant '{variant}'. Known: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[variant](**kwargs)
