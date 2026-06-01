"""Official-protocol evaluation driver."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .metrics import BinaryMetrics, average_binary_metrics, compute_binary_metrics


@dataclass
class PerSampleRecord:
    sample_id: str
    native_h: int
    native_w: int
    dice: float
    iou: float
    pred_area: int
    gt_area: int
    tp: int
    fp: int
    fn: int


@dataclass
class OfficialEvalResult:
    dice: float
    iou: float
    per_sample: List[BinaryMetrics]
    per_sample_records: List[PerSampleRecord]
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
    pyramid_transform: Any | None = None,
) -> OfficialEvalResult:
    """Run frozen SAM feature extraction + head forward pass + binary metrics.

    Prediction is upsampled to the GT resolution before thresholding so the
    headline ``direct_foreground_dice`` / ``direct_foreground_iou`` are computed
    on the GT raster.
    """

    head.eval()
    per_sample: List[BinaryMetrics] = []
    per_sample_records: List[PerSampleRecord] = []
    pred_res_seen, gt_res_seen = "?", "?"

    with torch.inference_mode():
        for si, sample in enumerate(samples):
            image: Image.Image = sample.image
            native_h, native_w = int(image.height), int(image.width)
            sample_id = str(getattr(sample, "sample_id", getattr(sample, "crop_name", str(si))))
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
            if pyramid_transform is not None:
                pyramid_np = pyramid_transform(sample_id, pyramid_np)
            pyramid_t = _pyramid_to_tensors(pyramid_np, device)
            logits = head(pyramid_t)  # [1, 1, h, w]
            pred_res_seen = f"{logits.shape[-2]}x{logits.shape[-1]}"

            logits_up = F.interpolate(
                logits, size=gt.shape, mode="bilinear", align_corners=False
            )
            pred = (torch.sigmoid(logits_up)[0, 0].cpu().numpy() > threshold)
            metrics = compute_binary_metrics(pred, gt)
            per_sample.append(metrics)

            tp = int(np.logical_and(pred, gt).sum())
            fp = int(np.logical_and(pred, ~gt).sum())
            fn = int(np.logical_and(~pred, gt).sum())
            per_sample_records.append(PerSampleRecord(
                sample_id=sample_id,
                native_h=native_h,
                native_w=native_w,
                dice=float(metrics.dice),
                iou=float(metrics.iou),
                pred_area=int(pred.sum()),
                gt_area=int(gt.sum()),
                tp=tp,
                fp=fp,
                fn=fn,
            ))

    avg = average_binary_metrics(per_sample)
    return OfficialEvalResult(
        dice=avg.dice,
        iou=avg.iou,
        per_sample=per_sample,
        per_sample_records=per_sample_records,
        prediction_resolution=pred_res_seen,
        gt_resolution=gt_res_seen,
    )


def write_per_sample_csv(path: Path, records: List[PerSampleRecord]) -> Path:
    """Write per-sample metrics to a CSV file.

    Columns: sample_id, native_h, native_w, dice, iou, pred_area, gt_area, tp, fp, fn
    """
    path = Path(path)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sample_id", "native_h", "native_w", "dice", "iou",
                         "pred_area", "gt_area", "tp", "fp", "fn"])
        for r in records:
            writer.writerow([
                r.sample_id, r.native_h, r.native_w,
                f"{r.dice:.6f}", f"{r.iou:.6f}",
                r.pred_area, r.gt_area, r.tp, r.fp, r.fn,
            ])
    return path
