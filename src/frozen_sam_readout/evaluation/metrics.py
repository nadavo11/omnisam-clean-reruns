"""Binary segmentation metrics. Copied verbatim from rwtd_sam3.eval.metrics.

This is the official scoring path. ``compute_binary_metrics`` produces the
``direct_foreground_dice`` / ``direct_foreground_iou`` reported in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class BinaryMetrics:
    """Binary segmentation metrics for one mask pair."""

    iou: float
    dice: float
    precision: float
    recall: float
    predicted_positive: int
    target_positive: int
    matched_positive: int

    def to_prefixed_dict(self, prefix: str) -> dict[str, float | int]:
        return {
            f"{prefix}_iou": self.iou,
            f"{prefix}_dice": self.dice,
            f"{prefix}_precision": self.precision,
            f"{prefix}_recall": self.recall,
            f"{prefix}_predicted_positive": self.predicted_positive,
            f"{prefix}_target_positive": self.target_positive,
            f"{prefix}_matched_positive": self.matched_positive,
        }


def compute_binary_metrics(prediction: np.ndarray, target: np.ndarray) -> BinaryMetrics:
    """Compute IoU, Dice, precision, and recall for two boolean masks."""

    pred = _validate_boolean_mask(prediction, "prediction")
    gt = _validate_boolean_mask(target, "target")
    _validate_same_shape(pred, gt)

    matched = int(np.logical_and(pred, gt).sum())
    pred_positive = int(pred.sum())
    target_positive = int(gt.sum())
    union = pred_positive + target_positive - matched

    if pred_positive == 0 and target_positive == 0:
        return BinaryMetrics(
            iou=1.0,
            dice=1.0,
            precision=1.0,
            recall=1.0,
            predicted_positive=0,
            target_positive=0,
            matched_positive=0,
        )

    precision = matched / pred_positive if pred_positive else 0.0
    recall = matched / target_positive if target_positive else 0.0
    dice = (2.0 * matched) / (pred_positive + target_positive) if (pred_positive + target_positive) else 1.0
    iou = matched / union if union else 1.0
    return BinaryMetrics(
        iou=float(iou),
        dice=float(dice),
        precision=float(precision),
        recall=float(recall),
        predicted_positive=pred_positive,
        target_positive=target_positive,
        matched_positive=matched,
    )


def average_binary_metrics(metrics_list: Iterable[BinaryMetrics]) -> BinaryMetrics:
    """Macro-average a sequence of metric groups across the scalar fields."""

    metrics = tuple(metrics_list)
    if not metrics:
        raise ValueError("Cannot average an empty metric list.")
    return BinaryMetrics(
        iou=float(np.mean([item.iou for item in metrics])),
        dice=float(np.mean([item.dice for item in metrics])),
        precision=float(np.mean([item.precision for item in metrics])),
        recall=float(np.mean([item.recall for item in metrics])),
        predicted_positive=int(sum(item.predicted_positive for item in metrics)),
        target_positive=int(sum(item.target_positive for item in metrics)),
        matched_positive=int(sum(item.matched_positive for item in metrics)),
    )


def _validate_boolean_mask(mask: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D mask, got shape {array.shape}.")
    return array


def _validate_same_shape(prediction: np.ndarray, target: np.ndarray) -> None:
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction and target masks must have the same shape, got "
            f"{prediction.shape} and {target.shape}."
        )
