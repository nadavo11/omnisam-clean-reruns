"""Synthetic-mask sanity checks for compute_binary_metrics."""

import numpy as np

from frozen_sam_readout.evaluation.metrics import compute_binary_metrics


def test_perfect_match():
    gt = np.zeros((10, 10), dtype=bool)
    gt[2:6, 2:6] = True
    m = compute_binary_metrics(gt.copy(), gt)
    assert m.dice == 1.0
    assert m.iou == 1.0
    assert m.precision == 1.0
    assert m.recall == 1.0


def test_disjoint_predictions():
    pred = np.zeros((10, 10), dtype=bool)
    gt = np.zeros((10, 10), dtype=bool)
    pred[0:3, 0:3] = True
    gt[5:8, 5:8] = True
    m = compute_binary_metrics(pred, gt)
    assert m.dice == 0.0
    assert m.iou == 0.0


def test_both_empty_returns_one():
    blank = np.zeros((5, 5), dtype=bool)
    m = compute_binary_metrics(blank, blank)
    assert m.dice == 1.0
    assert m.iou == 1.0


def test_half_overlap_dice():
    pred = np.zeros((10, 10), dtype=bool)
    gt = np.zeros((10, 10), dtype=bool)
    pred[0:4, 0:4] = True   # 16 px
    gt[2:6, 2:6] = True     # 16 px, overlap 4
    m = compute_binary_metrics(pred, gt)
    assert abs(m.dice - (2 * 4) / (16 + 16)) < 1e-9
    assert abs(m.iou - 4 / (16 + 16 - 4)) < 1e-9
