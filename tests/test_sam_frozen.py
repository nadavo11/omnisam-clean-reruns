"""Verify the freeze_utils helpers."""

import pytest
import torch
import torch.nn as nn

from frozen_sam_readout.sam.freeze_utils import assert_sam_frozen, count_trainable_params


def _make_mock_sam(trainable: bool = False) -> nn.Module:
    model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU(), nn.Conv2d(8, 4, 1))
    for p in model.parameters():
        p.requires_grad_(trainable)
    return model


def test_assert_sam_frozen_ok():
    model = _make_mock_sam(trainable=False)
    assert_sam_frozen(model)


def test_assert_sam_frozen_raises_on_trainable():
    model = _make_mock_sam(trainable=True)
    with pytest.raises(RuntimeError):
        assert_sam_frozen(model)


def test_count_trainable_params_total_zero_when_frozen():
    model = _make_mock_sam(trainable=False)
    counts = count_trainable_params(model)
    assert counts["total"] == 0


def test_count_trainable_params_total_positive_when_trainable():
    model = _make_mock_sam(trainable=True)
    counts = count_trainable_params(model)
    assert counts["total"] > 0
