#!/usr/bin/env python
"""Clean E1 ablation trainer with fixed validation IDs and W&B logging.

This script is intentionally narrow:
- MonuSeg only
- variants: a0_fpn2, a2_fpn2_fpn1_refine, a3_memory
- train on the official train split, evaluate on a fixed validation subset
- save local artifacts under ``outputs/<run_name>/``
- stream metrics and one fixed validation visualization to W&B when available

The clean validation subset is selected from a fixed sample-id manifest so that
all variants and seeds see the same validation examples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import sys
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import yaml

from frozen_sam_readout.data import iter_monuseg_binary_samples
from frozen_sam_readout.evaluation.official_eval import run_official_eval
from frozen_sam_readout.models.registry import build_head
from frozen_sam_readout.sam import build_frozen_sam_pyramid_extractor
from frozen_sam_readout.training import bce_dice_loss, save_checkpoint, train_one_epoch
from frozen_sam_readout.utils import load_yaml_config, set_seed, write_run_manifest
from frozen_sam_readout.visualization import compose_strip, binary_error_map, render_gt_overlay, render_single_mask_overlay

_DEFAULT_CHANNELS = {"f2_channels": 256, "f1_channels": 64, "f0_channels": 32}
_ALLOWED_VARIANTS = {"a0_fpn2", "a2_fpn2_fpn1_refine", "a3_memory"}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_fixed_sample_ids(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict) or "validation_sample_ids" not in payload:
        raise ValueError(f"{path} must contain validation_sample_ids.")
    ids = payload["validation_sample_ids"]
    if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
        raise ValueError(f"{path} validation_sample_ids must be a list of strings.")
    return payload


def _build_head(variant: str) -> torch.nn.Module:
    kwargs = dict(_DEFAULT_CHANNELS)
    if variant == "a0_fpn2":
        kwargs.pop("f1_channels", None)
        kwargs.pop("f0_channels", None)
    elif variant in {"a2_fpn2_fpn1_refine", "a3_memory"}:
        kwargs.pop("f0_channels", None)
    if variant == "a3_memory":
        kwargs["memory_tokens"] = 8
    return build_head(variant, **kwargs)


def _resize_image(image: Image.Image, resize_hw: tuple[int, int] | None) -> Image.Image:
    if resize_hw is None:
        return image
    return image.resize((resize_hw[1], resize_hw[0]), Image.BILINEAR)


def _streaming_feature_iter(
    extractor,
    samples: Iterable,
    resize_hw: tuple[int, int] | None,
) -> Iterator[dict]:
    for sample in samples:
        img = _resize_image(sample.image, resize_hw)
        _, pyramid_np = extractor.extract_sam_pyramid(img)
        pyramid_t = {
            name: torch.from_numpy(np.ascontiguousarray(f)).float()
            for name, f in pyramid_np.items()
        }
        target = torch.from_numpy(np.asarray(sample.texture_a_mask, dtype=np.float32))
        yield {"pyramid": pyramid_t, "target": target}


def _predict_sample(
    *,
    extractor,
    head: torch.nn.Module,
    sample,
    resize_hw: tuple[int, int] | None,
    device: torch.device,
    threshold: float,
) -> np.ndarray:
    image = _resize_image(sample.image, resize_hw)
    _, pyramid_np = extractor.extract_sam_pyramid(image)
    tensors = {
        name: torch.from_numpy(np.ascontiguousarray(arr)).float().unsqueeze(0).to(device)
        for name, arr in pyramid_np.items()
    }
    with torch.inference_mode():
        logits = head(tensors)
        logits_up = F.interpolate(
            logits, size=sample.texture_a_mask.shape, mode="bilinear", align_corners=False
        )
        return (torch.sigmoid(logits_up)[0, 0].cpu().numpy() > threshold).astype(bool)


def _make_fixed_visual(
    *,
    sample,
    pred: np.ndarray,
    out_dir: Path,
    epoch: int,
    variant: str,
    seed: int,
    eval_split: str,
    dice: float,
    iou: float,
) -> Path:
    sample_dir = out_dir / "visuals" / f"epoch_{epoch:03d}" / sample.crop_name
    sample_dir.mkdir(parents=True, exist_ok=True)
    image = sample.image.convert("RGB")
    gt = np.asarray(sample.texture_a_mask, dtype=bool)
    input_path = sample_dir / "input.png"
    gt_path = sample_dir / "gt_overlay.png"
    pred_path = sample_dir / "pred_overlay.png"
    error_path = sample_dir / "error_map.png"
    panel_path = sample_dir / "panel.png"

    image.save(input_path)
    gt_overlay = render_gt_overlay(image, gt)
    pred_overlay = render_single_mask_overlay(image, pred, color=(230, 56, 70))
    error = binary_error_map(pred, gt)
    gt_overlay.save(gt_path)
    pred_overlay.save(pred_path)
    Image.fromarray(error).save(error_path)
    panel = compose_strip(
        [
            np.asarray(image, dtype=np.uint8),
            np.asarray(gt_overlay, dtype=np.uint8),
            np.asarray(pred_overlay, dtype=np.uint8),
            error,
        ]
    )
    panel.save(panel_path)

    metadata = {
        "sample_id": sample.crop_name,
        "variant_id": variant,
        "seed": int(seed),
        "eval_split": eval_split,
        "dice": float(dice),
        "iou": float(iou),
        "epoch": int(epoch),
        "paths": {
            "input": str(input_path.relative_to(out_dir)),
            "gt_overlay": str(gt_path.relative_to(out_dir)),
            "pred_overlay": str(pred_path.relative_to(out_dir)),
            "error_map": str(error_path.relative_to(out_dir)),
            "panel": str(panel_path.relative_to(out_dir)),
        },
    }
    (sample_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return panel_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clean E1 ablation trainer for MoNuSeg.")
    parser.add_argument("--config", required=True, help="Method YAML (e.g. configs/official/method_a0_fpn2.yaml).")
    parser.add_argument("--protocol-config", default="configs/official/monuseg_strict_512.yaml")
    parser.add_argument("--variant", default=None, choices=sorted(_ALLOWED_VARIANTS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fixed-sample-ids", default=None)
    parser.add_argument("--sample-ids", dest="fixed_sample_ids", default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--wandb-project", default="frozen-sam-readout")
    parser.add_argument("--wandb-group", default="monuseg_E1_clean_ablation")
    parser.add_argument("--wandb-job-type", default="E1_ablation")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--allow-wandb-failure", action="store_true", default=True)
    args = parser.parse_args(argv)

    set_seed(args.seed)
    method_config = load_yaml_config(args.config)
    config = load_yaml_config(args.protocol_config)
    if str(config.get("dataset", {}).get("name")) != "monuseg":
        raise SystemExit("This clean E1 runner is currently monuseg-only.")

    sam_cfg = config.get("sam", {})
    if not sam_cfg.get("frozen", True):
        raise SystemExit("Clean E1 runs require a frozen SAM backbone.")
    if sam_cfg.get("use_prompt_encoder", False) or sam_cfg.get("use_mask_decoder", False):
        raise SystemExit("Clean E1 runs require SAM prompt/mask decoder disabled.")

    eval_cfg = config.get("evaluation", {})
    threshold = float(eval_cfg.get("threshold", 0.5))
    if threshold != 0.5:
        raise SystemExit("Clean E1 runs require threshold=0.5.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or out_dir.name
    split_manifest_path = out_dir / "split_manifest.json"
    if not args.fixed_sample_ids:
        raise SystemExit("--sample-ids/--fixed-sample-ids is required.")
    fixed_sample_ids_src = Path(args.fixed_sample_ids)
    fixed_sample_ids = _load_fixed_sample_ids(fixed_sample_ids_src)
    fixed_sample_ids_dst = out_dir / "fixed_sample_ids.json"
    fixed_sample_ids_dst.write_text(json.dumps(fixed_sample_ids, indent=2, sort_keys=True))

    train_samples = list(iter_monuseg_binary_samples(split="train"))
    val_ids = fixed_sample_ids["validation_sample_ids"]
    val_samples = [s for s in train_samples if str(s.crop_name) in set(val_ids)]
    train_samples = [s for s in train_samples if str(s.crop_name) not in set(val_ids)]
    if args.limit_train is not None:
        train_samples = train_samples[: int(args.limit_train)]
    if args.limit_val is not None:
        val_samples = val_samples[: int(args.limit_val)]
    if not val_samples:
        raise SystemExit("Validation sample selection is empty.")

    split_manifest = {
        "dataset": "monuseg",
        "train_split": "train",
        "val_split": "val",
        "selection_policy": fixed_sample_ids.get("selection_policy", "fixed"),
        "fixed_sample_ids_path": str(fixed_sample_ids_src.resolve()),
        "validation_sample_ids": [str(s.crop_name) for s in val_samples],
        "train_sample_ids": [str(s.crop_name) for s in train_samples],
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "source_train_count": len(list(iter_monuseg_binary_samples(split="train"))),
    }
    split_manifest_path.write_text(json.dumps(split_manifest, indent=2, sort_keys=True))

    variant = args.variant or str(method_config.get("model", {}).get("variant", ""))
    if variant not in _ALLOWED_VARIANTS:
        raise SystemExit(f"Could not infer a valid clean E1 variant from {args.config!r}: {variant!r}")

    (out_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "protocol": config,
                "method": method_config,
                "runtime": {
                    "run_name": run_name,
                    "variant": variant,
                    "seed": int(args.seed),
                },
            },
            sort_keys=False,
        )
    )

    write_run_manifest(
        out_dir,
        config={"protocol": config, "method": method_config},
        extras={
            "run_name": run_name,
            "seed": int(args.seed),
            "split_manifest_path": str(split_manifest_path),
            "fixed_sample_ids_path": str(fixed_sample_ids_dst),
            "wandb_project": args.wandb_project,
            "wandb_group": args.wandb_group,
            "wandb_job_type": args.wandb_job_type,
            "eval_split": "val",
        },
    )

    device = torch.device("cuda" if (args.device in {"auto", "cuda"} and torch.cuda.is_available()) else "cpu")
    sam_cfg = config["sam"]
    extractor = build_frozen_sam_pyramid_extractor(model_id=str(sam_cfg["model_id"]), device=args.device)
    head = _build_head(variant).to(device)
    training_cfg = config.get("training", {})
    optimizer = torch.optim.AdamW(
        [p for p in head.parameters() if p.requires_grad],
        lr=float(training_cfg.get("lr", 1e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )
    resize = config["dataset"].get("resize_before_sam")
    resize_hw = (int(resize["height"]), int(resize["width"])) if resize else None

    wandb_run = None
    wandb_url = "unavailable"
    try:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            job_type=args.wandb_job_type,
            name=run_name,
            tags=["monuseg", "clean_ablation", "E1", "heldout_val", "no_train_eval"],
            config={
                "run_name": run_name,
                "variant": variant,
                "seed": int(args.seed),
                "config_path": str(Path(args.config).resolve()),
                "split_manifest_path": str(split_manifest_path.resolve()),
                "fixed_sample_ids_path": str(fixed_sample_ids_dst.resolve()),
                "output_dir": str(out_dir.resolve()),
                "eval_split": "val",
            },
        )
        wandb_url = str(wandb_run.url)
    except Exception as exc:
        if not args.allow_wandb_failure:
            raise
        print(f"[warn] wandb init failed: {exc}", file=sys.stderr)

    (out_dir / "wandb_run_url.txt").write_text(wandb_url + "\n")
    (out_dir / "env_snapshot.txt").write_text(
        "\n".join(
            [
                f"python={sys.version}",
                f"platform={platform.platform()}",
                f"torch={torch.__version__}",
                f"cuda_available={torch.cuda.is_available()}",
                f"command={shlex.join(sys.argv)}",
            ]
        )
        + "\n"
    )

    best_val_dice = float("-inf")
    best_val_iou = float("-inf")
    best_ckpt = out_dir / "checkpoint.pt"
    latest_ckpt = out_dir / "checkpoint_last.pt"
    val_metrics_path = out_dir / "eval_metrics_val.json"
    test_metrics_path = out_dir / "eval_metrics_test.json"
    fixed_visual_sample = val_samples[0]

    for epoch in range(1, int(args.epochs) + 1):
        train_stream = _streaming_feature_iter(extractor, train_samples, resize_hw)
        train_stats = train_one_epoch(
            model=head,
            dataset=train_stream,
            optimizer=optimizer,
            loss_fn=bce_dice_loss,
            device=device,
        )

        val_result = run_official_eval(
            extractor=extractor,
            head=head,
            samples=val_samples,
            device=device,
            threshold=threshold,
            resize_before_sam_hw=resize_hw,
        )
        val_dice = float(val_result.dice)
        val_iou = float(val_result.iou)
        lr = float(optimizer.param_groups[0]["lr"])

        save_checkpoint(latest_ckpt, model=head, optimizer=optimizer, epoch=epoch)
        latest_sha = _sha256(latest_ckpt)
        if val_dice >= best_val_dice:
            best_val_dice = val_dice
            best_val_iou = val_iou
            save_checkpoint(best_ckpt, model=head, optimizer=optimizer, epoch=epoch)
        best_sha = _sha256(best_ckpt)

        pred = _predict_sample(
            extractor=extractor,
            head=head,
            sample=fixed_visual_sample,
            resize_hw=resize_hw,
            device=device,
            threshold=threshold,
        )
        panel_path = _make_fixed_visual(
            sample=fixed_visual_sample,
            pred=pred,
            out_dir=out_dir,
            epoch=epoch,
            variant=variant,
            seed=int(args.seed),
            eval_split="val",
            dice=val_dice,
            iou=val_iou,
        )

        metrics_payload = {
            "run_name": run_name,
            "dataset": "monuseg",
            "variant": variant,
            "seed": int(args.seed),
            "epoch": int(epoch),
            "train": {
                "loss": float(train_stats["loss"]),
                "lr": lr,
            },
            "val": {
                "dice": val_dice,
                "iou": val_iou,
                "eval_split": "val",
                "sample_id": fixed_visual_sample.crop_name,
            },
            "checkpoint_path": str(best_ckpt.resolve()),
            "checkpoint_sha256": best_sha,
            "latest_checkpoint_path": str(latest_ckpt.resolve()),
            "latest_checkpoint_sha256": latest_sha,
            "config_path": str(Path(args.config).resolve()),
            "split_manifest_path": str(split_manifest_path.resolve()),
            "fixed_sample_ids_path": str(fixed_sample_ids_dst.resolve()),
            "visual_path": str(panel_path.resolve()),
        }

        (out_dir / f"metrics_epoch_{epoch:03d}.json").write_text(json.dumps(metrics_payload, indent=2, sort_keys=True))

        val_summary = {
            "dataset": "monuseg",
            "variant": variant,
            "seed": int(args.seed),
            "epoch": int(epoch),
            "eval_split": "val",
            "checkpoint_path": str(best_ckpt.resolve()),
            "checkpoint_sha256": best_sha,
            "config_path": str(Path(args.config).resolve()),
            "split_manifest_path": str(split_manifest_path.resolve()),
            "fixed_sample_ids_path": str(fixed_sample_ids_dst.resolve()),
            "val_dice": val_dice,
            "val_iou": val_iou,
            "best_val_dice": best_val_dice,
            "best_val_iou": best_val_iou,
        }
        val_metrics_path.write_text(json.dumps(val_summary, indent=2, sort_keys=True))
        test_metrics_path.write_text(
            json.dumps(
                {
                    "dataset": "monuseg",
                    "variant": variant,
                    "seed": int(args.seed),
                    "eval_split": "test",
                    "status": "not_run",
                    "reason": "E1 validation-only clean ablation run",
                },
                indent=2,
                sort_keys=True,
            )
        )
        (out_dir / "checkpoint_sha256.txt").write_text(best_sha + "\n")

        if wandb_run is not None:
            try:
                import wandb

                wandb_run.summary.update(
                    {
                        "checkpoint_path": str(best_ckpt.resolve()),
                        "checkpoint_sha256": best_sha,
                        "eval_split": "val",
                        "config_path": str(Path(args.config).resolve()),
                        "split_manifest_path": str(split_manifest_path.resolve()),
                        "fixed_sample_ids_path": str(fixed_sample_ids_dst.resolve()),
                    }
                )
                wandb_run.log(
                    {
                        "epoch": int(epoch),
                        "train/loss": float(train_stats["loss"]),
                        "train/lr": lr,
                        "val/dice": val_dice,
                        "val/iou": val_iou,
                        "provenance/eval_split": "val",
                        "provenance/checkpoint_path": str(best_ckpt.resolve()),
                        "provenance/checkpoint_sha256": best_sha,
                        "provenance/config_path": str(Path(args.config).resolve()),
                        "provenance/split_manifest_path": str(split_manifest_path.resolve()),
                        "val/sample_segmentation": wandb.Image(
                            np.asarray(Image.open(panel_path)),
                            caption=(
                                f"{run_name} | sample={fixed_visual_sample.crop_name} | split=val "
                                f"| dice={val_dice:.4f} | iou={val_iou:.4f}"
                            ),
                        ),
                    }
                )
            except Exception as exc:
                print(f"[warn] wandb logging failed at epoch {epoch}: {exc}", file=sys.stderr)

        print(
            f"[{run_name}] epoch {epoch}/{args.epochs} "
            f"loss={train_stats['loss']:.4f} val_dice={val_dice:.4f} val_iou={val_iou:.4f}"
        )

    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass

    print(f"Saved best checkpoint: {best_ckpt}")
    print(f"Saved val metrics: {val_metrics_path}")
    print(f"Saved test stub: {test_metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
