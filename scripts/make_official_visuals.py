#!/usr/bin/env python
"""Run all variants on a small sample set and render the official comparison
grid: ``Input | GT | A0 | A2 | A3 | Final | Error``.

The script accepts one --checkpoint-dir with subfolders named after the
registry variant keys (e.g. ``a3_memory/stage1.pt``,
``final_staged/stage2.pt``). Missing variants are skipped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from frozen_sam_readout.data import build_dataset_iter
from frozen_sam_readout.models.registry import build_head
from frozen_sam_readout.sam import build_frozen_sam_pyramid_extractor
from frozen_sam_readout.training.checkpointing import load_checkpoint
from frozen_sam_readout.utils import load_yaml_config
from frozen_sam_readout.visualization import make_comparison_grid


_DEFAULT_CHANNELS = {"f2_channels": 256, "f1_channels": 64, "f0_channels": 32}

_VARIANTS = ["a0_fpn2", "a2_fpn2_fpn1_refine", "a3_memory", "final_staged"]


def _build_variant(variant: str, device) -> torch.nn.Module:
    kwargs = dict(_DEFAULT_CHANNELS)
    if variant == "a0_fpn2":
        kwargs.pop("f1_channels")
        kwargs.pop("f0_channels")
    elif variant in {"a2_fpn2_fpn1_refine", "a3_memory"}:
        kwargs.pop("f0_channels")
    return build_head(variant, **kwargs).to(device)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", required=False, default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    config = load_yaml_config(args.config)
    device = torch.device(
        "cuda" if (args.device in {"auto", "cuda"} and torch.cuda.is_available()) else "cpu"
    )
    extractor = build_frozen_sam_pyramid_extractor(
        model_id=str(config["sam"]["model_id"]), device=args.device
    )

    ds_cfg = config["dataset"]
    ds_kwargs: dict = {"split": ds_cfg.get("split", "test"), "limit": args.limit}
    if ds_cfg["name"] == "glas":
        ds_kwargs["dataset_root"] = ds_cfg["root_or_name"]
    samples = list(build_dataset_iter(ds_cfg["name"], **ds_kwargs))

    resize = ds_cfg.get("resize_before_sam")
    resize_hw = (int(resize["height"]), int(resize["width"])) if resize else None

    # Build all variants (load checkpoints if present).
    ckpt_root = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    heads = {}
    for variant in _VARIANTS:
        try:
            head = _build_variant(variant, device).eval()
        except Exception as exc:
            print(f"Skipping variant {variant}: {exc}")
            continue
        if ckpt_root is not None:
            candidate = next(iter((ckpt_root / variant).glob("*.pt")), None)
            if candidate is not None:
                load_checkpoint(candidate, model=head, map_location=str(device), strict=False)
        heads[variant] = head

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for sample in samples:
            img = sample.image
            inp = img.resize((resize_hw[1], resize_hw[0]), Image.BILINEAR) if resize_hw else img
            _, pyramid_np = extractor.extract_sam_pyramid(inp)
            pyramid_t = {
                k: torch.from_numpy(np.ascontiguousarray(v)).float().unsqueeze(0).to(device)
                for k, v in pyramid_np.items()
            }
            gt = np.asarray(sample.texture_a_mask, dtype=bool)
            predictions = {}
            for variant, head in heads.items():
                logits = head(pyramid_t)
                up = F.interpolate(logits, size=gt.shape, mode="bilinear", align_corners=False)
                predictions[variant] = (torch.sigmoid(up)[0, 0].cpu().numpy() > 0.5)
            out_path = out_dir / f"{sample.crop_name}.png"
            make_comparison_grid(
                image=img, gt=gt, predictions=predictions, output_path=out_path,
                variant_order=tuple(v for v in _VARIANTS if v in predictions),
            )
            print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
