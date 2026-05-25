#!/usr/bin/env python
"""Stage-2 training: load stage-1 A3 weights into the final staged head and
jointly train A3 + F0-learned-alpha residual for +40 epochs.

By default A3 is NOT frozen (per the official protocol). Use
``--freeze-a3`` to reproduce the freeze-A3 control ablation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from frozen_sam_readout.data import build_dataset_iter
from frozen_sam_readout.models.heads import StagedMultiscaleReadoutHead
from frozen_sam_readout.sam import build_frozen_sam_pyramid_extractor
from frozen_sam_readout.training import bce_dice_loss, load_checkpoint, save_checkpoint, train_one_epoch
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
    parser.add_argument("--variant", default="final_staged")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage1-checkpoint", default=None)
    parser.add_argument("--freeze-a3", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow-nonofficial", action="store_true")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    config = load_yaml_config(args.config)
    validate_official_guards(config, allow_nonofficial=args.allow_nonofficial)
    method_cfg = load_yaml_config(args.method) if args.method else {
        "model": {"variant": args.variant, "memory_tokens": 8, "f0": {"alpha_init": 0.05}}
    }

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

    model_cfg = method_cfg.get("model", {})
    head = StagedMultiscaleReadoutHead(
        **_DEFAULT_CHANNELS,
        decoder_dim=int(model_cfg.get("decoder_dim", 128)),
        memory_tokens=int(model_cfg.get("memory_tokens", 8)),
        alpha_init=float(model_cfg.get("f0", {}).get("alpha_init", 0.05)),
        final_layer_zero_init=bool(model_cfg.get("f0", {}).get("final_layer_zero_init", True)),
    ).to(device)

    if args.stage1_checkpoint:
        load_checkpoint(args.stage1_checkpoint, model=head.a3, map_location=str(device), strict=False)
        print(f"Warm-started A3 from {args.stage1_checkpoint}")

    head.set_f0_enabled(True)
    if args.freeze_a3:
        head.freeze_a3(True)
        print("Stage-2 control: A3 frozen.")

    training_cfg = config.get("training", {})
    optimizer = torch.optim.AdamW(
        [p for p in head.parameters() if p.requires_grad],
        lr=float(training_cfg.get("lr", 1e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )

    epochs = int(training_cfg.get("epochs_stage2", 40))
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
        alpha = float(head.alpha.detach().cpu()) if head.alpha is not None else float("nan")
        print(f"[stage2] epoch {epoch + 1}/{epochs}  loss={stats['loss']:.4f}  alpha={alpha:.4f}")

    ckpt_path = out_dir / "stage2.pt"
    save_checkpoint(ckpt_path, model=head, optimizer=optimizer, epoch=epochs)
    print(f"Saved {ckpt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
