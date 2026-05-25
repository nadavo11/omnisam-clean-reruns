"""Helpers for asserting SAM frozenness and counting trainable parameters."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


def assert_sam_frozen(model: Any, *, sam_attr_candidates: Iterable[str] = ("sam", "image_encoder", "vision_encoder", "backbone")) -> None:
    """Raise ``RuntimeError`` if any SAM parameter has ``requires_grad=True``.

    The function looks at the model itself and any common SAM submodule names.
    For a plain SAM model (no wrapper), it iterates ``model.parameters()``.
    """

    trainable: List[str] = []
    seen: set[int] = set()
    targets = [model]
    for name in sam_attr_candidates:
        sub = getattr(model, name, None)
        if sub is not None:
            targets.append(sub)
    for target in targets:
        if id(target) in seen:
            continue
        seen.add(id(target))
        if not hasattr(target, "named_parameters"):
            continue
        for pname, param in target.named_parameters():
            if param.requires_grad:
                trainable.append(pname)
    if trainable:
        raise RuntimeError(
            f"SAM contract violation: {len(trainable)} parameters are trainable; "
            f"first offenders: {trainable[:5]}"
        )


def count_trainable_params(model: Any) -> Dict[str, int]:
    """Return ``{submodule_name: trainable_param_count}`` plus a ``total`` key."""

    counts: Dict[str, int] = {}
    total = 0
    if hasattr(model, "named_children"):
        for child_name, child in model.named_children():
            sub_total = sum(
                int(p.numel()) for p in child.parameters() if p.requires_grad
            )
            counts[child_name] = sub_total
            total += sub_total
    else:
        total = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    counts["total"] = total
    return counts
