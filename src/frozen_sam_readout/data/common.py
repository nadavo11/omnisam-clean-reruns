"""Shared dataset helpers (boundary derivation, validation)."""

from __future__ import annotations

import numpy as np


def boundary_from_region_masks(
    texture_a_mask: np.ndarray, texture_b_mask: np.ndarray
) -> np.ndarray:
    """Derive boundary mask using 8-neighbour adjacency between two regions."""

    mask_a = np.asarray(texture_a_mask, dtype=bool)
    mask_b = np.asarray(texture_b_mask, dtype=bool)
    if mask_a.shape != mask_b.shape:
        raise ValueError(
            f"texture_a_mask and texture_b_mask shape mismatch: {mask_a.shape} vs {mask_b.shape}."
        )
    boundary = np.zeros_like(mask_a, dtype=bool)
    height, width = mask_a.shape
    padded_a = np.pad(mask_a, pad_width=1, mode="constant", constant_values=False)
    padded_b = np.pad(mask_b, pad_width=1, mode="constant", constant_values=False)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            shifted_b = padded_b[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]
            shifted_a = padded_a[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]
            boundary |= (mask_a & shifted_b) | (mask_b & shifted_a)
    return boundary
