"""Error-map rendering for binary prediction vs GT."""

from __future__ import annotations

import numpy as np


def binary_error_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Return an RGB uint8 image: green=TP, blue=FP, red=FN, black=TN."""

    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch {pred.shape} vs {gt.shape}")
    out = np.zeros((*pred.shape, 3), dtype=np.uint8)
    tp = pred & gt
    fp = pred & ~gt
    fn = ~pred & gt
    out[tp] = (0, 200, 0)
    out[fp] = (50, 100, 255)
    out[fn] = (220, 0, 0)
    return out
