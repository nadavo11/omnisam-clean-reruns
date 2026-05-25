"""Official comparison-grid visualizer.

Columns: Input | GT | A0 | A2 | A3 | Final | Error
Where each prediction is a thresholded binary mask. ``Error`` corresponds to
the ``Final`` head vs GT (TP=green, FP=blue, FN=red, TN=black).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np
from PIL import Image

from .error_maps import binary_error_map
from .grids import compose_row


def _mask_to_panel(mask: np.ndarray) -> np.ndarray:
    return (np.asarray(mask, dtype=bool).astype(np.uint8) * 255)


def make_comparison_grid(
    *,
    image: Image.Image,
    gt: np.ndarray,
    predictions: Dict[str, np.ndarray],
    output_path: str | Path,
    variant_order: Sequence[str] = ("a0_fpn2", "a2_fpn2_fpn1_refine", "a3_memory", "final_staged"),
    titles: Sequence[str] = ("Input", "GT", "A0", "A2", "A3", "Final", "Error"),
) -> Path:
    panels = [np.asarray(image.convert("RGB")), _mask_to_panel(gt)]
    for name in variant_order:
        panels.append(_mask_to_panel(predictions[name]))
    final = predictions[variant_order[-1]]
    panels.append(binary_error_map(final, gt))
    grid = compose_row(panels, titles=titles)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    return output_path
