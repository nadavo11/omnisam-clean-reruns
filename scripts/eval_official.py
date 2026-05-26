#!/usr/bin/env python
"""Official eval driver: load config, validate guards, run frozen extraction +
head forward pass, compute ``direct_foreground_dice`` / ``direct_foreground_iou``,
write ``official_metrics.json`` with the locked schema.
"""

from __future__ import annotations

import argparse
import hashlib
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from frozen_sam_readout.data import build_dataset_iter
from frozen_sam_readout.evaluation.official_eval import run_official_eval, write_per_sample_csv
from frozen_sam_readout.evaluation.prediction_io import (
    build_official_metrics_payload,
    write_official_metrics_json,
)
from frozen_sam_readout.models.compat.source_loader import load_source_checkpoint_head
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


def _resolve_official_eval_split(config: dict) -> str:
    eval_cfg = config.get("evaluation", {})
    return str(eval_cfg.get("split", "test"))


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


def _checkpoint_is_source_format(path: str) -> bool:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return isinstance(payload, dict) and "head_state_dict" in payload


def _checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    eval_split = _resolve_official_eval_split(config)

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
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        checkpoint_sha256 = _checkpoint_sha256(checkpoint_path)
        if _checkpoint_is_source_format(args.checkpoint):
            head = load_source_checkpoint_head(args.checkpoint, device=device)
        else:
            head = _build_head_from_config(method_config).to(device)
            load_checkpoint(args.checkpoint, model=head, map_location=str(device))
    else:
        head = _build_head_from_config(method_config).to(device)
        checkpoint_sha256 = None

    # Build samples.
    ds_cfg = config["dataset"]
    ds_name = str(ds_cfg["name"])
    ds_kwargs: dict = {"split": eval_split, "limit": args.limit}
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
        checkpoint_sha256=checkpoint_sha256,
        config_path=str(Path(args.config).resolve()),
        method_config_path=str(Path(args.method_config).resolve()) if args.method_config else None,
        evaluator_script=str(Path(__file__).resolve()),
        evaluator_command=shlex.join(sys.argv),
        created_at=datetime.now(timezone.utc).isoformat(),
        metric_schema_version=2,
        provenance_status="canonical" if not args.allow_nonofficial else "noncanonical",
        seed=int(args.seed),
        eval_split=eval_split,
        dice=result.dice,
        iou=result.iou,
        threshold=float(config["evaluation"].get("threshold", 0.5)),
        prediction_resolution=result.prediction_resolution,
        gt_resolution=result.gt_resolution,
        paper_headline_safe=not args.allow_nonofficial,
    )
    out_path = write_official_metrics_json(out_dir / "official_metrics.json", payload)
    csv_path = write_per_sample_csv(out_dir / "per_sample_metrics.csv", result.per_sample_records)
    print(f"Wrote {out_path}")
    print(f"Wrote {csv_path}")
    print(
        f"  direct_foreground_dice={result.dice:.4f}  "
        f"direct_foreground_iou={result.iou:.4f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
