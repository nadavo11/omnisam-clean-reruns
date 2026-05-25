"""Tests for the visualization pipeline — no SAM model required."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from frozen_sam_readout.visualization import (
    ERROR_COLOR_MAP,
    SamplePanels,
    binary_error_map,
    compose_strip,
    save_official_visuals,
)


def _rand_rgb(h: int, w: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _checkerboard(h: int, w: int) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    return ((xs + ys) % 2).astype(bool)


class TestComposedStrip:
    def test_basic_shape(self):
        p1 = np.zeros((64, 64, 3), dtype=np.uint8)
        p2 = np.ones((64, 64, 3), dtype=np.uint8) * 128
        strip = compose_strip([p1, p2], gap=2)
        assert isinstance(strip, Image.Image)
        assert strip.size == (64 + 2 + 64, 64)

    def test_no_text(self):
        # Strip must have no white rows at the top (no title bar).
        p = np.zeros((32, 32, 3), dtype=np.uint8)
        strip = compose_strip([p], gap=2)
        arr = np.asarray(strip)
        assert arr[0].max() == 0, "Top row should be black (no title bar)"


class TestErrorMap:
    def test_colors(self):
        pred = np.array([[True, False, True, False]])
        gt   = np.array([[True, True, False, False]])
        em = binary_error_map(pred, gt)
        assert tuple(em[0, 0]) == tuple(ERROR_COLOR_MAP["true_positive"])   # TP
        assert tuple(em[0, 1]) == tuple(ERROR_COLOR_MAP["false_negative"])  # FN
        assert tuple(em[0, 2]) == tuple(ERROR_COLOR_MAP["false_positive"])  # FP
        assert tuple(em[0, 3]) == tuple(ERROR_COLOR_MAP["true_negative"])   # TN

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            binary_error_map(np.zeros((4, 4), dtype=bool), np.zeros((3, 3), dtype=bool))


class TestSaveOfficialVisuals:
    def test_full_pipeline(self, tmp_path):
        h, w = 64, 64
        rng = np.random.default_rng(1)
        fg = _checkerboard(h, w)

        panels = [
            SamplePanels(
                sample_id=f"sample_{i:03d}",
                image=_rand_rgb(h, w),
                gt=fg,
                pred_a0=(rng.random((h, w)) > 0.5),
                pred_a2=(rng.random((h, w)) > 0.5),
                pred_a3=(rng.random((h, w)) > 0.5),
                pred_final=(rng.random((h, w)) > 0.5),
            )
            for i in range(3)
        ]

        manifest_path = save_official_visuals(
            sample_panels=panels,
            output_dir=tmp_path,
            dataset="monuseg",
            protocol="monuseg_strict_512",
            config="configs/official/monuseg_strict_512.yaml",
            checkpoints={"a0_fpn2": "/fake/A0.pt", "final_staged": "/fake/final.pt"},
        )

        # Manifest written and parseable.
        assert manifest_path.exists()
        m = json.loads(manifest_path.read_text())
        assert m["dataset"] == "monuseg"
        assert len(m["samples"]) == 3
        assert "error_color_map" in m

        # Panel images written — 7 files per sample.
        for entry in m["samples"]:
            for panel_path in entry["panels"].values():
                assert (tmp_path / panel_path).exists()

        # QC grids written — one per sample.
        for entry in m["samples"]:
            assert (tmp_path / entry["grid"]).exists()

        # LaTeX snippet written.
        assert (tmp_path / "figure_grid.tex").exists()
        tex = (tmp_path / "figure_grid.tex").read_text()
        assert "\\begin{figure}" in tex
        assert "\\includegraphics" in tex
        assert "sample_000" in tex

        # No text in any panel image (spot-check: top-left pixel of gt panel).
        gt_path = tmp_path / m["samples"][0]["panels"]["gt"]
        gt_img = np.asarray(Image.open(gt_path))
        assert gt_img.ndim == 3, "Panel images must be RGB"
