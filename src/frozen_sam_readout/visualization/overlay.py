"""Native mask-overlay renderers — ported verbatim from rwtd_sam3/utils/visualization.py.

Rendering contract (identical to source repo):
  - Prediction mask: colour α=0.42 + white contour α=0.95 over the RGB image.
  - GT mask:         neutral grey (200,200,200) α=0.42 + white contour α=0.95.
  - All blends use float32 linear interpolation, clipped to uint8.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Primitives (ported verbatim)
# ---------------------------------------------------------------------------

def _blend_mask(
    image_array: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    color_array = np.array(color, dtype=np.float32)
    output = image_array.astype(np.float32)
    boolean_mask = np.asarray(mask, dtype=bool)
    output[boolean_mask] = (1.0 - alpha) * output[boolean_mask] + alpha * color_array
    return np.clip(output, 0, 255).astype(np.uint8)


def _erode_mask(mask: np.ndarray) -> np.ndarray:
    boolean_mask = np.asarray(mask, dtype=bool)
    height, width = boolean_mask.shape
    padded = np.pad(boolean_mask, pad_width=1, mode="constant", constant_values=False)
    neighbours = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            neighbours.append(padded[1 + dy: 1 + dy + height, 1 + dx: 1 + dx + width])
    return np.logical_and.reduce(neighbours)


def _mask_outline(mask: np.ndarray) -> np.ndarray:
    boolean_mask = np.asarray(mask, dtype=bool)
    return np.logical_and(boolean_mask, np.logical_not(_erode_mask(boolean_mask)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_single_mask_overlay(
    image: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int],
) -> Image.Image:
    """Overlay one binary mask: colour fill α=0.42 + white outline α=0.95."""
    base = np.asarray(image.convert("RGB"), dtype=np.uint8)
    composed = base.copy()
    boolean_mask = np.asarray(mask, dtype=bool)
    composed = _blend_mask(composed, boolean_mask, color=color, alpha=0.42)
    composed = _blend_mask(composed, _mask_outline(boolean_mask), color=(255, 255, 255), alpha=0.95)
    return Image.fromarray(composed, mode="RGB")


def render_gt_overlay(
    image: Image.Image,
    mask: np.ndarray,
) -> Image.Image:
    """GT overlay: neutral grey fill α=0.42 + white outline α=0.95."""
    return render_single_mask_overlay(image, mask, color=(200, 200, 200))
