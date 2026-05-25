"""Grid composition — clean, text-free panel strips for QC and wandb."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from PIL import Image


def _to_rgb(arr: np.ndarray) -> np.ndarray:
    """Coerce any array to uint8 RGB [H,W,3]."""
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr


def compose_strip(
    panels: Sequence[np.ndarray],
    gap: int = 2,
    gap_color: tuple[int, int, int] = (40, 40, 40),
) -> Image.Image:
    """Horizontally concatenate panels with a thin dark gap. No text.

    All panels are padded to the same height before concatenation.

    Args:
        panels: Sequence of H×W or H×W×3 uint8/float arrays.
        gap: Width in pixels of the separator between panels.
        gap_color: RGB color of the separator.

    Returns:
        PIL Image: the composed strip.
    """
    rgb = [_to_rgb(p) for p in panels]
    h = max(p.shape[0] for p in rgb)

    pieces: list[np.ndarray] = []
    sep = np.full((h, gap, 3), gap_color, dtype=np.uint8)

    for i, panel in enumerate(rgb):
        # Pad shorter panels to common height with black rows at the bottom.
        if panel.shape[0] < h:
            pad = np.zeros((h - panel.shape[0], panel.shape[1], 3), dtype=np.uint8)
            panel = np.vstack([panel, pad])
        pieces.append(panel)
        if i < len(rgb) - 1:
            pieces.append(sep)

    return Image.fromarray(np.concatenate(pieces, axis=1))
