"""Error-map rendering for binary prediction vs GT.

Color convention (matches source qual_grids color scheme):
  True positive  → (180, 180, 180)  subtle grey overlay
  False positive → (0,   200,   0)  green  — model over-segments
  False negative → (200,   0,   0)  red    — model under-segments
  True negative  → (  0,   0,   0)  black  — correct background
"""

from __future__ import annotations

import numpy as np

# Canonical color map — reference this dict in visuals_manifest.json.
ERROR_COLOR_MAP: dict[str, tuple[int, int, int]] = {
    "true_positive":  (180, 180, 180),
    "false_positive": (  0, 200,   0),
    "false_negative": (200,   0,   0),
    "true_negative":  (  0,   0,   0),
}


def binary_error_map(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    color_map: dict[str, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Return an RGB uint8 error map for a binary prediction against GT.

    Args:
        pred: Boolean or 0/1 array [H, W] — model prediction.
        gt:   Boolean or 0/1 array [H, W] — ground truth.
        color_map: Override ``ERROR_COLOR_MAP`` if provided.

    Returns:
        uint8 [H, W, 3] RGB array.
    """
    cm = color_map if color_map is not None else ERROR_COLOR_MAP
    pred = np.asarray(pred, dtype=bool)
    gt   = np.asarray(gt,   dtype=bool)
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}")

    out = np.zeros((*pred.shape, 3), dtype=np.uint8)
    out[pred &  gt] = cm["true_positive"]
    out[pred & ~gt] = cm["false_positive"]
    out[~pred & gt] = cm["false_negative"]
    # true_negative stays black (already zero)
    return out
