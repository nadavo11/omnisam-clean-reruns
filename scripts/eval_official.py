#!/usr/bin/env python
"""Official eval driver: load config, validate guards, run frozen extraction +
head forward pass, compute ``direct_foreground_dice`` / ``direct_foreground_iou``,
write ``official_metrics.json`` with the locked schema.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from frozen_sam_readout.data import build_dataset_iter
from frozen_sam_readout.evaluation.official_eval import run_official_eval
from frozen_sam_readout.evaluation.prediction_io import (
    build_official_metrics_payload,
    write_official_metrics_json,
)
from frozen_sam_readout.models.registry import build_head
from frozen_sam_readout.sam import build_frozen_sam_pyramid_extractor
from frozen_sam_readout.training.checkpointing import load_checkpoint
from frozen_sam_readout.utils import (
    load_yaml_config,
    set_seed,
    validate_official_guards,
    write_run_manifest,
)


# Default SAM channels used when the user does not provide a method config that
# carries explicit channel counts. These are the standard SAM-2 hiera-l shapes.
_DEFAULT_CHANNELS = {"f2_channels": 256, "f1_channels": 64, "f0_channels": 32}


def _build_head_from_config(method_config: dict) -> torch.nn.Module:
    model_cfg = method_config.get("model", {})
    variant = str(model_cfg.get("variant", "final_staged"))
    kwargs = dict(_DEFAULT_CHANNELS)
    kwargs["decoder_dim"] = int(model_cfg.get("decoder_dim", 128))
    if variant in {"a3_memory", "a5_all_at_once", "final_staged"}:
        kwargs["memory_tokens"] = int(model_cfg.get("memory_tokens", 8))
    if variant in {"a5_all_at_once", "final_staged"}:
        f0 = model_cfg.get("f0", {})
        kwargs["alpha_init"] = float(f0.get("alpha_init", 0.05))
    if variant == "final_staged":
        kwargs["projection_dim"] = int(model_cfg.get("projection_dim", 128))
        kwargs["final_layer_zero_init"] = bool(
            method_config.get("model", {}).get("f0", {}).get("final_layer_zero_init", True)
        )
    if variant == "a0_fpn2":
        kwargs.pop("f1_channels", None)
        kwargs.pop("f0_channels", None)
    if variant in {"a2_fpn2_fpn1_refine", "a3_memory"}:
        kwargs.pop("f0_channels", None)
    return build_head(variant, **kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Official eval for the frozen-SAM readout.")
    parser.add_argument("--config", required=True, help="Path to dataset/protocol YAML config.")
    parser.add_argument("--method-config", default=None, help="Optional model/method YAML.")
    parser.add_argument("--checkpoint", default=None, help="Path to head checkpoint .pt file.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--allow-nonofficial", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    set_seed(args.seed)
    config = load_yaml_config(args.config)
    validate_official_guards(config, allow_nonofficial=args.allow_nonofficial)

    method_config = (
        load_yaml_config(args.method_config) if args.method_config else {"model": {"variant": "final_staged"}}
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_run_manifest(out_dir, config={"protocol": config, "method": method_config})

    # Build extractor.
    sam_cfg = config["sam"]
    extractor = build_frozen_sam_pyramid_extractor(
        model_id=str(sam_cfg["model_id"]),
        device=args.device,
    )

    # Build head and (optionally) load checkpoint.
    device = torch.device(
        "cuda" if (args.device in {"auto", "cuda"} and torch.cuda.is_available()) else "cpu"
    )
    head = _build_head_from_config(method_config).to(device)
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model=head, map_location=str(device))

    # Build samples.
    ds_cfg = config["dataset"]
    ds_name = str(ds_cfg["name"])
    ds_kwargs: dict = {"split": ds_cfg.get("split", "test"), "limit": args.limit}
    if ds_name == "glas":
        ds_kwargs["dataset_root"] = ds_cfg["root_or_name"]
    samples = build_dataset_iter(ds_name, **ds_kwargs)

    resize = ds_cfg.get("resize_before_sam")
    resize_hw = (
        (int(resize["height"]), int(resize["width"])) if resize else None
    )

    # Run.
    result = run_official_eval(
        extractor=extractor,
        head=head,
        samples=samples,
        device=device,
        threshold=float(config["evaluation"].get("threshold", 0.5)),
        resize_before_sam_hw=resize_hw,
    )

    payload = build_official_metrics_payload(
        dataset=ds_name,
        protocol=Path(args.config).stem,
        variant=str(method_config.get("model", {}).get("variant", "final_staged")),
        checkpoint=str(args.checkpoint or ""),
        seed=int(args.seed),
        dice=result.dice,
        iou=result.iou,
        threshold=float(config["evaluation"].get("threshold", 0.5)),
        prediction_resolution=result.prediction_resolution,
        gt_resolution=result.gt_resolution,
        paper_headline_safe=not args.allow_nonofficial,
    )
    out_path = write_official_metrics_json(out_dir / "official_metrics.json", payload)
    print(f"Wrote {out_path}")
    print(
        f"  direct_foreground_dice={result.dice:.4f}  "
        f"direct_foreground_iou={result.iou:.4f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
