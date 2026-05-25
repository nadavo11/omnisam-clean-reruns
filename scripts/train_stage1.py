#!/usr/bin/env python
"""Stage-1 training: train A3 (or any pre-F0 variant) on cached frozen features.

The script expects the user to provide a feature-cache iterator (a list of
dicts with ``pyramid`` and ``target`` keys). For brevity the script ships with
a thin loader that streams images through the frozen SAM extractor on the fly;
swap in a cached-tensor dataset for serious training runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from frozen_sam_readout.data import build_dataset_iter
from frozen_sam_readout.models.registry import build_head
from frozen_sam_readout.sam import build_frozen_sam_pyramid_extractor
from frozen_sam_readout.training import bce_dice_loss, save_checkpoint, train_one_epoch
from frozen_sam_readout.utils import (
    load_yaml_config,
    set_seed,
    validate_official_guards,
    write_run_manifest,
)


_DEFAULT_CHANNELS = {"f2_channels": 256, "f1_channels": 64, "f0_channels": 32}


def _streaming_feature_iter(extractor, samples, resize_hw) -> Iterator[dict]:
    from PIL import Image

    for sample in samples:
        img = sample.image
        if resize_hw is not None:
            img = img.resize((resize_hw[1], resize_hw[0]), Image.BILINEAR)
        _, pyramid_np = extractor.extract_sam_pyramid(img)
        pyramid_t = {
            name: torch.from_numpy(np.ascontiguousarray(f)).float()
            for name, f in pyramid_np.items()
        }
        target = torch.from_numpy(np.asarray(sample.texture_a_mask, dtype=np.float32))
        yield {"pyramid": pyramid_t, "target": target}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", default=None)
    parser.add_argument("--variant", default="a3_memory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow-nonofficial", action="store_true")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    config = load_yaml_config(args.config)
    validate_official_guards(config, allow_nonofficial=args.allow_nonofficial)
    method_cfg = load_yaml_config(args.method) if args.method else {"model": {"variant": args.variant}}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_run_manifest(out_dir, config={"protocol": config, "method": method_cfg})

    device = torch.device(
        "cuda" if (args.device in {"auto", "cuda"} and torch.cuda.is_available()) else "cpu"
    )

    sam_cfg = config["sam"]
    extractor = build_frozen_sam_pyramid_extractor(
        model_id=str(sam_cfg["model_id"]), device=args.device
    )

    variant = str(method_cfg.get("model", {}).get("variant", args.variant))
    kwargs = dict(_DEFAULT_CHANNELS)
    if variant == "a0_fpn2":
        kwargs.pop("f1_channels")
        kwargs.pop("f0_channels")
    elif variant in {"a2_fpn2_fpn1_refine", "a3_memory"}:
        kwargs.pop("f0_channels")
    head = build_head(variant, **kwargs).to(device)

    training_cfg = config.get("training", {})
    optimizer = torch.optim.AdamW(
        [p for p in head.parameters() if p.requires_grad],
        lr=float(training_cfg.get("lr", 1e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )

    epochs = int(training_cfg.get("epochs_stage1", 40))
    ds_cfg = config["dataset"]
    ds_name = str(ds_cfg["name"])
    ds_kwargs: dict = {"split": "train", "limit": args.limit}
    if ds_name == "glas":
        ds_kwargs["dataset_root"] = ds_cfg["root_or_name"]
    resize = ds_cfg.get("resize_before_sam")
    resize_hw = (int(resize["height"]), int(resize["width"])) if resize else None

    for epoch in range(epochs):
        samples = build_dataset_iter(ds_name, **ds_kwargs)
        stream = _streaming_feature_iter(extractor, samples, resize_hw)
        stats = train_one_epoch(
            model=head, dataset=stream, optimizer=optimizer, loss_fn=bce_dice_loss, device=device,
        )
        print(f"[stage1] epoch {epoch + 1}/{epochs}  loss={stats['loss']:.4f}")

    ckpt_path = out_dir / "stage1.pt"
    save_checkpoint(ckpt_path, model=head, optimizer=optimizer, epoch=epochs)
    print(f"Saved {ckpt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
