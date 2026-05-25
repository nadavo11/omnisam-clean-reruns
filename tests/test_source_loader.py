"""Tests for source-checkpoint compatibility loader."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from frozen_sam_readout.models.compat.source_loader import (
    _SourceA0Head,
    _SourceA2Head,
    _SourceA3Head,
    _SourceFinalStagedHead,
    _DECODER_DIM,
)


def _dummy_pyramid(h: int = 72, w: int = 72, device: str = "cpu") -> dict[str, torch.Tensor]:
    return {
        "fpn_2": torch.randn(1, 256, h, w, device=device),
        "fpn_1": torch.randn(1, 256, h * 2, w * 2, device=device),
        "fpn_0": torch.randn(1, 256, h * 4, w * 4, device=device),
    }


class TestSourceA0Head:
    def test_forward_shape(self):
        model = _SourceA0Head().eval()
        pyr = _dummy_pyramid()
        with torch.inference_mode():
            out = model(pyr)
        assert out.shape == (1, 1, 72, 72)

    def test_key_names(self):
        model = _SourceA0Head()
        keys = set(model.state_dict().keys())
        assert "projections.fpn_2.weight" in keys
        assert "projections.fpn_2.bias" in keys
        assert "decoder.0.weight" in keys
        assert "decoder.0.bias" in keys
        assert "classifier.weight" in keys

    def test_conv_has_bias(self):
        model = _SourceA0Head()
        assert model.decoder[0].bias is not None, "Source decoder conv must have bias"


class TestSourceA2Head:
    def test_forward_shape(self):
        model = _SourceA2Head().eval()
        pyr = _dummy_pyramid()
        with torch.inference_mode():
            out = model(pyr)
        assert out.ndim == 4

    def test_key_names(self):
        model = _SourceA2Head()
        keys = set(model.state_dict().keys())
        assert "residual_scale" in keys
        assert "coarse_head.projections.fpn_2.weight" in keys
        assert "fine_projection.weight" in keys
        assert "residual_classifier.weight" in keys


class TestSourceA3Head:
    def test_forward_shape(self):
        model = _SourceA3Head().eval()
        pyr = _dummy_pyramid()
        with torch.inference_mode():
            out = model(pyr)
        assert out.ndim == 4

    def test_key_names(self):
        model = _SourceA3Head()
        keys = set(model.state_dict().keys())
        assert "memory_tokens" in keys
        assert "primary_projection.weight" in keys
        assert "skip_projections.fpn_1.weight" in keys
        assert "blocks.0.query_norm.weight" in keys


class TestSourceFinalStagedHead:
    def test_forward_shape(self):
        model = _SourceFinalStagedHead().eval()
        pyr = _dummy_pyramid()
        with torch.inference_mode():
            out = model(pyr)
        assert out.ndim == 4

    def test_key_names(self):
        model = _SourceFinalStagedHead()
        keys = set(model.state_dict().keys())
        assert "raw_alpha" in keys
        assert "f0_projection.weight" in keys
        assert "skip_projection.weight" in keys  # no 's' in final_staged


class TestLoadSourceCheckpointHead:
    """Integration test — skipped unless source checkpoint files are available."""

    CKPT_A0 = "/storage/nada/outputs/fixed_resize_enhancement/monuseg_A0_fpn2_d128_512_e40/checkpoint.pt"

    @pytest.mark.skipif(
        not __import__("pathlib").Path(CKPT_A0).exists(),
        reason="Cluster checkpoint not available locally",
    )
    def test_load_a0(self):
        from frozen_sam_readout.models.compat.source_loader import load_source_checkpoint_head
        model = load_source_checkpoint_head(self.CKPT_A0, device="cpu")
        assert isinstance(model, nn.Module)
        pyr = _dummy_pyramid(72, 72)
        with torch.inference_mode():
            out = model(pyr)
        assert out.shape[-1] == 72
