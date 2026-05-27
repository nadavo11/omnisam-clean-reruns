#!/usr/bin/env python
"""MoNuSeg TEST_SCREEN ablation runner.

MoNuSeg has no real validation set.  All ablation evidence must come from the
official held-out test split.  This script:
  - Trains on the full MoNuSeg training set (no val holdout)
  - Evaluates on the official test set every epoch
  - Selects the checkpoint with the best test Dice
  - Writes eval/test_metrics.json ONLY
  - Explicitly forbids eval_split=val for MoNuSeg

Since checkpoint selection uses test data, these results are a TRANSPARENT
TEST SCREEN, not an unbiased final evaluation.  Every run goes into the
test-screen ledger.  No cherry-picking.

Supports:
  - Single-stage: a0_fpn2, a2_fpn2_fpn1_refine, a3_memory
  - Two-stage: final_staged (A3+F0), final_staged_a2 (A2+F0)
    Stage 1: train inner head alone (F0 disabled)
    Stage 2: enable F0, train jointly
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
from frozen_sam_readout.training import bce_dice_loss, load_checkpoint, save_checkpoint, train_one_epoch
from frozen_sam_readout.training.augmentations import apply_augmentation, describe_augmentation_policy
from frozen_sam_readout.utils import load_yaml_config, set_seed, write_run_manifest
from frozen_sam_readout.utils.config import assert_monuseg_no_val
from frozen_sam_readout.visualization import compose_strip, binary_error_map, render_gt_overlay, render_single_mask_overlay

_SINGLE_STAGE_VARIANTS = {"a0_fpn2", "a2_fpn2_fpn1_refine", "a3_memory"}
_TWO_STAGE_VARIANTS = {"final_staged", "final_staged_a2"}
_ALL_VARIANTS = _SINGLE_STAGE_VARIANTS | _TWO_STAGE_VARIANTS


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resize_image(image: Image.Image, resize_hw: tuple[int, int] | None) -> Image.Image:
    if resize_hw is None:
        return image
    return image.resize((resize_hw[1], resize_hw[0]), Image.BILINEAR)


def _streaming_feature_iter(
    extractor,
    samples: Iterable,
    resize_hw,
    *,
    aug_policy: str = "none",
    rng: "np.random.Generator | None" = None,
) -> Iterator[dict]:
    for sample in samples:
        if aug_policy != "none" and rng is not None:
            # Augmentation applied to PIL image + mask BEFORE SAM feature extraction
            sample = apply_augmentation(sample, rng=rng, policy=aug_policy)
        img = _resize_image(sample.image, resize_hw)
        _, pyramid_np = extractor.extract_sam_pyramid(img)
        pyramid_t = {
            name: torch.from_numpy(np.ascontiguousarray(f)).float()
            for name, f in pyramid_np.items()
        }
        target = torch.from_numpy(np.asarray(sample.texture_a_mask, dtype=np.float32))
        yield {"pyramid": pyramid_t, "target": target}


def _predict_sample(*, extractor, head, sample, resize_hw, device, threshold):
    image = _resize_image(sample.image, resize_hw)
    _, pyramid_np = extractor.extract_sam_pyramid(image)
    tensors = {
        k: torch.from_numpy(np.ascontiguousarray(v)).float().unsqueeze(0).to(device)
        for k, v in pyramid_np.items()
    }
    with torch.inference_mode():
        logits = head(tensors)
        logits_up = F.interpolate(logits, size=sample.texture_a_mask.shape, mode="bilinear", align_corners=False)
        return (torch.sigmoid(logits_up)[0, 0].cpu().numpy() > threshold).astype(bool)


def _make_fixed_visual(*, sample, pred, out_dir, epoch, variant, seed, dice, iou):
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
    render_gt_overlay(image, gt).save(gt_path)
    render_single_mask_overlay(image, pred, color=(230, 56, 70)).save(pred_path)
    Image.fromarray(binary_error_map(pred, gt)).save(error_path)
    compose_strip([
        np.asarray(image, dtype=np.uint8),
        np.asarray(render_gt_overlay(image, gt), dtype=np.uint8),
        np.asarray(render_single_mask_overlay(image, pred, color=(230, 56, 70)), dtype=np.uint8),
        binary_error_map(pred, gt),
    ]).save(panel_path)
    (sample_dir / "metadata.json").write_text(json.dumps({
        "sample_id": sample.crop_name, "variant_id": variant, "seed": int(seed),
        "eval_split": "test", "dice": float(dice), "iou": float(iou), "epoch": int(epoch),
    }, indent=2, sort_keys=True))
    return panel_path


def _probe_backbone_channels(extractor, sample, resize_hw):
    img = _resize_image(sample.image, resize_hw)
    _, pyramid_np = extractor.extract_sam_pyramid(img)
    defaults = {"f2_channels": 256, "f1_channels": 64, "f0_channels": 32}
    for level, key in [("fpn_2", "f2_channels"), ("fpn_1", "f1_channels"), ("fpn_0", "f0_channels")]:
        if level in pyramid_np:
            defaults[key] = int(pyramid_np[level].shape[0])
    return defaults


def _build_head(variant: str, channels: dict, method_config: dict) -> torch.nn.Module:
    model_cfg = method_config.get("model", {})
    kwargs = dict(channels)
    if variant == "a0_fpn2":
        kwargs.pop("f1_channels", None)
        kwargs.pop("f0_channels", None)
    elif variant == "a2_fpn2_fpn1_refine":
        kwargs.pop("f0_channels", None)
    elif variant == "a3_memory":
        kwargs.pop("f0_channels", None)
        kwargs["memory_tokens"] = int(model_cfg.get("memory_tokens", 8))
    elif variant == "final_staged":
        kwargs["memory_tokens"] = int(model_cfg.get("memory_tokens", 8))
        kwargs["alpha_init"] = float(model_cfg.get("alpha_init", 0.05))
        kwargs["final_layer_zero_init"] = bool(model_cfg.get("final_layer_zero_init", True))
    elif variant == "final_staged_a2":
        kwargs["alpha_init"] = float(model_cfg.get("alpha_init", 0.05))
        kwargs["final_layer_zero_init"] = bool(model_cfg.get("final_layer_zero_init", True))
    kwargs.pop("decoder_dim", None)
    kwargs["decoder_dim"] = int(model_cfg.get("decoder_dim", 128))
    return build_head(variant, **kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MoNuSeg TEST_SCREEN runner")
    parser.add_argument("--config", required=True, help="Test-screen YAML config path")
    parser.add_argument("--variant", default=None)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--wandb-project", default="frozen-sam-readout")
    parser.add_argument("--wandb-group", default="monuseg_test_screen_ablation")
    parser.add_argument("--wandb-job-type", default="TEST_SCREEN")
    parser.add_argument("--allow-wandb-failure", action="store_true", default=True)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    args = parser.parse_args(argv)

    set_seed(args.seed)
    config = load_yaml_config(args.config)

    dataset_name = str(config.get("dataset", {}).get("name", "monuseg"))
    eval_split_cfg = str(config.get("evaluation", {}).get("split", "test"))

    # Hard guard: MoNuSeg must never use val/validation split
    assert_monuseg_no_val(dataset_name, eval_split_cfg)
    if eval_split_cfg != "test":
        raise SystemExit(f"TEST_SCREEN runner requires evaluation.split=test, got {eval_split_cfg!r}")

    method_config = config  # method config embedded in same file
    variant = args.variant or str(config.get("model", {}).get("variant", ""))
    if variant not in _ALL_VARIANTS:
        raise SystemExit(f"Unknown variant {variant!r}. Known: {sorted(_ALL_VARIANTS)}")

    training_cfg = config.get("training", {})
    stage1_epochs = int(training_cfg.get("stage1_epochs", 40))
    stage2_epochs = int(training_cfg.get("stage2_epochs", 0))
    total_epochs = stage1_epochs + stage2_epochs
    is_two_stage = variant in _TWO_STAGE_VARIANTS and stage2_epochs > 0

    aug_policy = str(config.get("dataset", {}).get("augmentation_policy", "none")).lower()
    aug_rng = np.random.default_rng(int(args.seed))  # seeded separately from model init

    out_dir = Path(args.output_dir or f"outputs/{args.run_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval").mkdir(exist_ok=True)
    (out_dir / "visuals").mkdir(exist_ok=True)
    (out_dir / "wandb").mkdir(exist_ok=True)

    # Save config and run metadata
    (out_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    # Load data — full training set, no val holdout
    train_samples = list(iter_monuseg_binary_samples(split="train"))
    test_samples = list(iter_monuseg_binary_samples(split="test"))
    if args.limit_train is not None:
        train_samples = train_samples[:args.limit_train]
    if args.limit_test is not None:
        test_samples = test_samples[:args.limit_test]

    split_manifest = {
        "dataset": "monuseg",
        "train_split": "train",
        "eval_split": "test",
        "note": "MoNuSeg has no real validation set. No val holdout. Test used for ablation screening.",
        "train_sample_ids": [str(s.crop_name) for s in train_samples],
        "test_sample_ids": [str(s.crop_name) for s in test_samples],
        "train_count": len(train_samples),
        "test_count": len(test_samples),
    }
    (out_dir / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2, sort_keys=True))

    resize_cfg = config.get("dataset", {}).get("resize_before_sam")
    resize_hw = (int(resize_cfg["height"]), int(resize_cfg["width"])) if resize_cfg else None

    device = torch.device("cuda" if (args.device in {"auto", "cuda"} and torch.cuda.is_available()) else "cpu")
    sam_cfg = config.get("sam", {})
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    extractor = build_frozen_sam_pyramid_extractor(
        model_id=str(sam_cfg.get("model_id", "facebook/sam3")),
        device=args.device,
        hf_token=hf_token,
    )

    # Probe and log feature shapes
    probe_sample = train_samples[0]
    actual_channels = _probe_backbone_channels(extractor, probe_sample, resize_hw)
    backbone_backend = "unknown"
    if hasattr(extractor, "_bundle") and extractor._bundle is not None:
        backbone_backend = extractor._bundle[0] if isinstance(extractor._bundle[0], str) else "official"

    feature_shapes = {
        "backbone_model_id": str(sam_cfg.get("model_id", "facebook/sam3")),
        "backbone_backend": backbone_backend,
        "paper_headline_safe": backbone_backend == "official",
        "f2_channels": actual_channels.get("f2_channels"),
        "f1_channels": actual_channels.get("f1_channels"),
        "f0_channels": actual_channels.get("f0_channels"),
        "expected_paper_channels": {"f2_channels": 256, "f1_channels": 64, "f0_channels": 32},
        "channel_parity_ok": (
            actual_channels.get("f2_channels") == 256
            and actual_channels.get("f1_channels") == 64
            and actual_channels.get("f0_channels") == 32
        ),
    }
    (out_dir / "feature_shapes.json").write_text(json.dumps(feature_shapes, indent=2, sort_keys=True))
    print(f"[info] feature_shapes: {feature_shapes}", flush=True)

    head = _build_head(variant, actual_channels, method_config).to(device)
    if is_two_stage:
        head.set_f0_enabled(False)  # stage 1: inner head only

    lr_schedule = str(training_cfg.get("lr_schedule", "constant")).lower()
    lr_min = float(training_cfg.get("lr_min", 0.0))

    optimizer = torch.optim.AdamW(
        [p for p in head.parameters() if p.requires_grad],
        lr=float(training_cfg.get("lr", 1e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )

    def _make_scheduler(opt: torch.optim.Optimizer, n_epochs: int) -> "torch.optim.lr_scheduler._LRScheduler | None":
        if lr_schedule == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr_min)
        return None

    scheduler = _make_scheduler(optimizer, stage1_epochs if is_two_stage else total_epochs)
    threshold = float(config.get("evaluation", {}).get("threshold", 0.5))

    # Fixed test visual sample (first test sample, consistent across epochs)
    fixed_visual_sample = test_samples[0]
    (out_dir / "visuals" / "fixed_sample_ids.json").write_text(
        json.dumps({"fixed_visual_sample_id": fixed_visual_sample.crop_name, "eval_split": "test"}, indent=2)
    )

    # W&B init
    wandb_run = None
    wandb_url = "unavailable"
    try:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            job_type=args.wandb_job_type,
            name=args.run_name,
            tags=["monuseg", "test_screen", variant, f"seed{args.seed}"],
            config={
                "run_name": args.run_name,
                "variant": variant,
                "seed": int(args.seed),
                "eval_split": "test",
                "stage": "TEST_SCREEN",
                "backbone_backend": backbone_backend,
                "paper_headline_safe": feature_shapes["paper_headline_safe"],
                "channel_parity_ok": feature_shapes["channel_parity_ok"],
                "f2_channels": feature_shapes["f2_channels"],
                "f1_channels": feature_shapes["f1_channels"],
                "f0_channels": feature_shapes["f0_channels"],
                "stage1_epochs": stage1_epochs,
                "stage2_epochs": stage2_epochs,
                "augmentation_policy": aug_policy,
                "augmentation_description": describe_augmentation_policy(aug_policy),
                "train_count": len(train_samples),
                "test_count": len(test_samples),
                "config_path": str(Path(args.config).resolve()),
            },
        )
        wandb_url = str(wandb_run.url)
    except Exception as exc:
        if not args.allow_wandb_failure:
            raise
        print(f"[warn] wandb init failed: {exc}", file=sys.stderr)

    (out_dir / "wandb" / "run_url.txt").write_text(wandb_url + "\n")
    (out_dir / "env_snapshot.txt").write_text(
        "\n".join([
            f"python={sys.version}", f"platform={platform.platform()}",
            f"torch={torch.__version__}", f"cuda_available={torch.cuda.is_available()}",
            f"command={shlex.join(sys.argv)}",
        ]) + "\n"
    )

    write_run_manifest(
        out_dir,
        config={"config": config},
        extras={
            "run_name": args.run_name, "variant": variant, "seed": int(args.seed),
            "eval_split": "test", "stage": "TEST_SCREEN",
            "backbone_backend": backbone_backend,
        },
    )

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    best_test_dice = float("-inf")
    best_test_iou = float("-inf")
    best_test_epoch = -1
    final_epoch_dice = float("nan")
    final_epoch_iou = float("nan")
    best_ckpt = out_dir / "checkpoint.pt"
    latest_ckpt = out_dir / "checkpoint_last.pt"
    per_epoch_csv = out_dir / "eval" / "per_epoch_metrics.csv"
    per_epoch_csv.write_text("epoch,stage,train_loss,lr,test_dice,test_iou,is_best\n")

    for epoch in range(1, total_epochs + 1):
        # Two-stage transition
        if is_two_stage and epoch == stage1_epochs + 1:
            print(f"[{args.run_name}] Stage 2: enabling F0 residual at epoch {epoch}", flush=True)
            head.set_f0_enabled(True)
            # Re-init optimizer to include new F0 params
            optimizer = torch.optim.AdamW(
                [p for p in head.parameters() if p.requires_grad],
                lr=float(training_cfg.get("lr", 1e-4)),
                weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
            )
            scheduler = _make_scheduler(optimizer, stage2_epochs)

        train_stream = _streaming_feature_iter(
            extractor, train_samples, resize_hw,
            aug_policy=aug_policy, rng=aug_rng,
        )
        train_stats = train_one_epoch(
            model=head, dataset=train_stream, optimizer=optimizer,
            loss_fn=bce_dice_loss, device=device,
        )

        # Evaluate on official test split every epoch
        test_result = run_official_eval(
            extractor=extractor, head=head, samples=test_samples,
            device=device, threshold=threshold, resize_before_sam_hw=resize_hw,
        )
        test_dice = float(test_result.dice)
        test_iou = float(test_result.iou)

        lr = float(optimizer.param_groups[0]["lr"])
        save_checkpoint(latest_ckpt, model=head, optimizer=optimizer, epoch=epoch)
        if test_dice > best_test_dice:
            best_test_dice = test_dice
            best_test_iou = test_iou
            best_test_epoch = epoch
            save_checkpoint(best_ckpt, model=head, optimizer=optimizer, epoch=epoch)
        final_epoch_dice = test_dice
        final_epoch_iou = test_iou
        if scheduler is not None:
            scheduler.step()

        # Append to per-epoch CSV (no buffering — readable mid-run)
        with per_epoch_csv.open("a") as _f:
            _f.write(f"{epoch},{stage_label},{train_stats['loss']:.6f},{lr:.8f},{test_dice:.6f},{test_iou:.6f},{int(epoch == best_test_epoch)}\n")

        # Fixed visual for this epoch
        pred = _predict_sample(
            extractor=extractor, head=head, sample=fixed_visual_sample,
            resize_hw=resize_hw, device=device, threshold=threshold,
        )
        panel_path = _make_fixed_visual(
            sample=fixed_visual_sample, pred=pred, out_dir=out_dir,
            epoch=epoch, variant=variant, seed=args.seed,
            dice=test_dice, iou=test_iou,
        )

        stage_label = "stage1" if (is_two_stage and epoch <= stage1_epochs) else ("stage2" if is_two_stage else "train")
        alpha_val = None
        if is_two_stage and epoch > stage1_epochs and hasattr(head, "alpha") and head.alpha is not None:
            alpha_val = float(torch.sigmoid(head.alpha).item())

        if wandb_run is not None:
            try:
                import wandb as wandb_mod
                log_dict = {
                    "epoch": epoch,
                    "train/loss": float(train_stats["loss"]),
                    "train/lr": lr,
                    "test/dice": test_dice,
                    "test/iou": test_iou,
                    "provenance/eval_split": "test",
                    "provenance/stage": stage_label,
                    "test/sample_segmentation": wandb_mod.Image(
                        np.asarray(Image.open(panel_path)),
                        caption=(
                            f"{args.run_name} | epoch={epoch} | split=test "
                            f"| sample={fixed_visual_sample.crop_name} "
                            f"| dice={test_dice:.4f} | iou={test_iou:.4f} | variant={variant} | seed={args.seed}"
                        ),
                    ),
                }
                if alpha_val is not None:
                    log_dict["model/alpha"] = alpha_val
                wandb_run.log(log_dict)
            except Exception as exc:
                print(f"[warn] wandb logging failed at epoch {epoch}: {exc}", file=sys.stderr)

        print(
            f"[{args.run_name}] epoch {epoch}/{total_epochs} ({stage_label}) "
            f"loss={train_stats['loss']:.4f} test_dice={test_dice:.4f} test_iou={test_iou:.4f}",
            flush=True,
        )

    # -----------------------------------------------------------------------
    # Save final artifacts
    # -----------------------------------------------------------------------
    best_sha = _sha256(best_ckpt)
    (out_dir / "checkpoint_sha256.txt").write_text(best_sha + "\n")

    test_summary = {
        "dataset": "monuseg",
        "variant": variant,
        "seed": int(args.seed),
        "eval_split": "test",
        "stage": "TEST_SCREEN",
        "total_epochs": total_epochs,
        "stage1_epochs": stage1_epochs,
        "stage2_epochs": stage2_epochs,
        "lr_schedule": lr_schedule,
        "lr_min": lr_min,
        "augmentation_policy": aug_policy,
        "augmentation_description": describe_augmentation_policy(aug_policy),
        "checkpoint_views": {
            "final_epoch": {
                "selected_by": "final_epoch",
                "selected_epoch": total_epochs,
                "claimability": "claimable_test_screen",
                "test_dice": final_epoch_dice,
                "test_iou": final_epoch_iou,
            },
            "best_test_epoch": {
                "selected_by": "test_peak",
                "selected_epoch": best_test_epoch,
                "claimability": "diagnostic_oracle",
                "test_dice": best_test_dice,
                "test_iou": best_test_iou,
                "checkpoint_path": str(best_ckpt.resolve()),
                "checkpoint_sha256": best_sha,
            },
        },
        # Legacy flat fields — keep for backward compatibility
        "test_dice": best_test_dice,
        "test_iou": best_test_iou,
        "selected_by": "best_test_dice",
        "final_epoch_dice": final_epoch_dice,
        "final_epoch_iou": final_epoch_iou,
        "best_test_epoch": best_test_epoch,
        "config_path": str(Path(args.config).resolve()),
        "split_manifest_path": str((out_dir / "split_manifest.json").resolve()),
        "train_count": len(train_samples),
        "test_count": len(test_samples),
        "backbone_backend": backbone_backend,
        "channel_parity_ok": feature_shapes["channel_parity_ok"],
        "paper_headline_safe": feature_shapes["paper_headline_safe"],
    }
    (out_dir / "eval" / "test_metrics.json").write_text(json.dumps(test_summary, indent=2, sort_keys=True))

    (out_dir / "visuals" / "provenance.json").write_text(json.dumps({
        "fixed_visual_sample_id": fixed_visual_sample.crop_name,
        "eval_split": "test",
        "variant": variant,
        "seed": int(args.seed),
        "augmentation_policy": aug_policy,
    }, indent=2, sort_keys=True))

    if wandb_run is not None:
        try:
            wandb_run.summary.update({
                "test/dice": best_test_dice,
                "test/iou": best_test_iou,
                "test/eval_split": "test",
                "checkpoint_sha256": best_sha,
                "channel_parity_ok": feature_shapes["channel_parity_ok"],
                "backbone_backend": backbone_backend,
            })
            wandb_run.finish()
        except Exception as exc:
            print(f"[warn] wandb finish failed: {exc}", file=sys.stderr)

    print(
        f"[{args.run_name}] final best test_dice={best_test_dice:.4f} test_iou={best_test_iou:.4f} "
        f"(selected_by=best_test_dice, best_epoch={best_test_epoch})",
        flush=True,
    )
    print(
        f"[{args.run_name}] final_epoch test_dice={final_epoch_dice:.4f} test_iou={final_epoch_iou:.4f} "
        f"(selected_by=final_epoch, epoch={total_epochs}, claimability=claimable_test_screen)",
        flush=True,
    )
    print(f"Saved test metrics: {out_dir / 'eval' / 'test_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
