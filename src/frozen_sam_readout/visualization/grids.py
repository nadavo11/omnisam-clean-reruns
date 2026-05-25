"""Grid composition utilities."""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _to_rgb(arr: np.ndarray) -> np.ndarray:
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr


def compose_row(
    panels: Sequence[np.ndarray],
    titles: Optional[Sequence[str]] = None,
    pad: int = 4,
    title_height: int = 18,
) -> Image.Image:
    """Horizontally concatenate ``panels`` (numpy arrays) with optional titles."""

    rgb_panels = [_to_rgb(p) for p in panels]
    heights = [p.shape[0] for p in rgb_panels]
    widths = [p.shape[1] for p in rgb_panels]
    h = max(heights)
    total_w = sum(widths) + pad * (len(rgb_panels) - 1)
    total_h = h + (title_height if titles else 0)
    canvas = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))
    x = 0
    for idx, panel in enumerate(rgb_panels):
        img = Image.fromarray(panel)
        canvas.paste(img, (x, title_height if titles else 0))
        x += widths[idx] + pad
    if titles:
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        x = 0
        for idx, title in enumerate(titles):
            draw.text((x + 2, 2), str(title), fill=(0, 0, 0), font=font)
            x += widths[idx] + pad
    return canvas
