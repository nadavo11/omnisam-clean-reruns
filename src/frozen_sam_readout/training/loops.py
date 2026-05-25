"""Minimal training loop over a cached frozen-feature dataset.

The expected per-sample dict has keys::

    {
        "pyramid": {"fpn_2": [C2,H2,W2], "fpn_1": [C1,H1,W1], "fpn_0": [C0,H0,W0]},
        "target":  [H_target, W_target] (bool/float),
    }
"""

from __future__ import annotations

from typing import Callable, Iterable, Mapping

import torch
import torch.nn.functional as F


def _to_device(pyramid: Mapping[str, torch.Tensor], device: torch.device) -> dict:
    return {k: v.unsqueeze(0).to(device, non_blocking=True) for k, v in pyramid.items()}


def train_one_epoch(
    *,
    model: torch.nn.Module,
    dataset: Iterable[dict],
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
) -> dict[str, float]:
    """Run one epoch and return ``{'loss': avg_loss}``."""

    model.train()
    total_loss = 0.0
    count = 0
    for sample in dataset:
        pyramid = _to_device(sample["pyramid"], device)
        target = sample["target"].to(device).float()
        logits = model(pyramid)  # [1, 1, h, w]
        target_resized = F.interpolate(
            target[None, None, ...], size=logits.shape[-2:], mode="bilinear", align_corners=False
        )
        loss = loss_fn(logits, target_resized)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        count += 1
    return {"loss": total_loss / max(count, 1)}
