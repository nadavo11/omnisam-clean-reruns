"""Official-protocol evaluation driver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .metrics import BinaryMetrics, average_binary_metrics, compute_binary_metrics


@dataclass
class OfficialEvalResult:
    dice: float
    iou: float
    per_sample: List[BinaryMetrics]
    prediction_resolution: str
    gt_resolution: str


def _pyramid_to_tensors(pyramid: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        name: torch.from_numpy(np.ascontiguousarray(fmap))
        .float()
        .unsqueeze(0)
        .to(device, non_blocking=True)
        for name, fmap in pyramid.items()
    }


def run_official_eval(
    *,
    extractor: Any,
    head: torch.nn.Module,
    samples: Iterable[Any],
    device: torch.device,
    threshold: float = 0.5,
    resize_before_sam_hw: tuple[int, int] | None = None,
) -> OfficialEvalResult:
    """Run frozen SAM feature extraction + head forward pass + binary metrics.

    Prediction is upsampled to the GT resolution before thresholding so the
    headline ``direct_foreground_dice`` / ``direct_foreground_iou`` are computed
    on the GT raster.
    """

    head.eval()
    per_sample: List[BinaryMetrics] = []
    pred_res_seen, gt_res_seen = "?", "?"

    with torch.inference_mode():
        for sample in samples:
            image: Image.Image = sample.image
            gt = np.asarray(sample.texture_a_mask, dtype=bool)
            gt_res_seen = f"{gt.shape[0]}x{gt.shape[1]}"

            if resize_before_sam_hw is not None:
                inp = image.resize(
                    (int(resize_before_sam_hw[1]), int(resize_before_sam_hw[0])),
                    Image.BILINEAR,
                )
            else:
                inp = image

            _, pyramid_np = extractor.extract_sam_pyramid(inp)
            pyramid_t = _pyramid_to_tensors(pyramid_np, device)
            logits = head(pyramid_t)  # [1, 1, h, w]
            pred_res_seen = f"{logits.shape[-2]}x{logits.shape[-1]}"

            logits_up = F.interpolate(
                logits, size=gt.shape, mode="bilinear", align_corners=False
            )
            pred = (torch.sigmoid(logits_up)[0, 0].cpu().numpy() > threshold)
            per_sample.append(compute_binary_metrics(pred, gt))

    avg = average_binary_metrics(per_sample)
    return OfficialEvalResult(
        dice=avg.dice,
        iou=avg.iou,
        per_sample=per_sample,
        prediction_resolution=pred_res_seen,
        gt_resolution=gt_res_seen,
    )
