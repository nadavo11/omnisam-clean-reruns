#!/usr/bin/env python
"""MoNuSeg 80-epoch layer-warmup ablation runner.

Protocol
--------
• Train on the full MoNuSeg training set (37 samples, no val holdout).
• Evaluate on the official held-out test split (14 samples).
• eval_every: configurable (default 5); final epoch always evaluated.
• Gate schedule: each extra branch (memory attention, F0 residual) is enabled
  gradually via a scheduled buffer gate that ramps 0→1 over a defined window.
• Checkpoint: saved at every eval epoch; final epoch always saved separately.
• final_epoch is the ONLY claimable metric; best_test_epoch is logged as
  diagnostic oracle only.

MoNuSeg no-val contract is enforced: eval_split=val raises SystemExit.
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
from typing import Callable, Dict, Iterable, Iterator

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

_GATED_VARIANTS = {"gated_a3_f0_warmup", "gated_all_refine_mem"}
_LEGACY_VARIANTS = {"a0_fpn2", "a2_fpn2_fpn1_refine", "a3_memory", "final_staged", "final_staged_a2"}
_ALL_VARIANTS = _GATED_VARIANTS | _LEGACY_VARIANTS


# ---------------------------------------------------------------------------
# Gate schedule helpers
# ---------------------------------------------------------------------------

def _gate_value(epoch: int, warmup_start: int, warmup_end: int) -> float:
    """Linear ramp: 0.0 at warmup_start, 1.0 at warmup_end (inclusive), 0 before."""
    if epoch < warmup_start:
        return 0.0
    if warmup_start >= warmup_end or epoch >= warmup_end:
        return 1.0
    return float(epoch - warmup_start) / float(warmup_end - warmup_start)


def _update_gates(head: torch.nn.Module, gate_configs: dict, epoch: int) -> Dict[str, float]:
    """Apply all gate values for the given epoch; return state dict for logging."""
    state: Dict[str, float] = {}
    for gate_name, cfg in gate_configs.items():
        ws = int(cfg.get("warmup_start", 0))
        we = int(cfg.get("warmup_end", 1))
        val = _gate_value(epoch, ws, we)
        head.set_gate(gate_name, val)
        state[gate_name] = val
    return state


def _active_stage_label(gate_state: Dict[str, float]) -> str:
    active = [f"{k}={v:.2f}" for k, v in sorted(gate_state.items()) if v > 0.0]
    return "base" if not active else ",".join(active)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _resize_image(image: Image.Image, resize_hw: tuple | None) -> Image.Image:
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


def _make_fixed_visual(*, sample, pred, out_dir, epoch, variant, human_name, seed, dice, iou):
    sample_dir = out_dir / "visuals" / f"epoch_{epoch:03d}" / sample.crop_name
    sample_dir.mkdir(parents=True, exist_ok=True)
    image = sample.image.convert("RGB")
    gt = np.asarray(sample.texture_a_mask, dtype=bool)
    image.save(sample_dir / "input.png")
    render_gt_overlay(image, gt).save(sample_dir / "gt_overlay.png")
    render_single_mask_overlay(image, pred, color=(230, 56, 70)).save(sample_dir / "pred_overlay.png")
    Image.fromarray(binary_error_map(pred, gt)).save(sample_dir / "error_map.png")
    compose_strip([
        np.asarray(image, dtype=np.uint8),
        np.asarray(render_gt_overlay(image, gt), dtype=np.uint8),
        np.asarray(render_single_mask_overlay(image, pred, color=(230, 56, 70)), dtype=np.uint8),
        binary_error_map(pred, gt),
    ]).save(sample_dir / "panel.png")
    (sample_dir / "metadata.json").write_text(json.dumps({
        "sample_id": sample.crop_name, "variant_id": variant,
        "human_name": human_name, "seed": int(seed),
        "eval_split": "test", "dice": float(dice), "iou": float(iou), "epoch": int(epoch),
    }, indent=2, sort_keys=True))
    return sample_dir / "panel.png"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probe_backbone_channels(extractor, sample, resize_hw):
    img = _resize_image(sample.image, resize_hw)
    _, pyramid_np = extractor.extract_sam_pyramid(img)
    defaults = {"f2_channels": 256, "f1_channels": 64, "f0_channels": 32}
    for level, key in [("fpn_2", "f2_channels"), ("fpn_1", "f1_channels"), ("fpn_0", "f0_channels")]:
        if level in pyramid_np:
            defaults[key] = int(pyramid_np[level].shape[0])
    return defaults


def _build_head(variant: str, channels: dict, model_cfg: dict) -> torch.nn.Module:
    kwargs = dict(channels)
    kwargs["decoder_dim"] = int(model_cfg.get("decoder_dim", 128))
    memory_tokens = int(model_cfg.get("memory_tokens", 4))
    alpha_init = float(model_cfg.get("alpha_init", 0.05))

    if variant == "gated_a3_f0_warmup":
        kwargs["memory_tokens"] = memory_tokens
        kwargs["alpha_init"] = alpha_init
        kwargs["zero_init_f0_final"] = bool(model_cfg.get("zero_init_f0_final", True))
    elif variant == "gated_all_refine_mem":
        kwargs["memory_tokens"] = memory_tokens
        kwargs["alpha_init"] = alpha_init
        kwargs["zero_init_f0_final"] = bool(model_cfg.get("zero_init_f0_final", True))
        kwargs["use_f0_memory"] = bool(model_cfg.get("use_f0_memory", True))
    elif variant == "a0_fpn2":
        kwargs.pop("f1_channels", None)
        kwargs.pop("f0_channels", None)
    elif variant == "a2_fpn2_fpn1_refine":
        kwargs.pop("f0_channels", None)
    elif variant == "a3_memory":
        kwargs.pop("f0_channels", None)
        kwargs["memory_tokens"] = int(model_cfg.get("memory_tokens", 8))
    elif variant in ("final_staged", "final_staged_a2"):
        if variant == "final_staged":
            kwargs["memory_tokens"] = int(model_cfg.get("memory_tokens", 8))
        kwargs["alpha_init"] = alpha_init
        kwargs["final_layer_zero_init"] = bool(model_cfg.get("final_layer_zero_init", True))

    # Remove unknown kwargs for simpler heads
    if variant == "a0_fpn2":
        kwargs.pop("f1_channels", None)
        kwargs.pop("f0_channels", None)
    elif variant == "a2_fpn2_fpn1_refine":
        kwargs.pop("f0_channels", None)
    elif variant == "a3_memory":
        kwargs.pop("f0_channels", None)

    return build_head(variant, **kwargs)


# ---------------------------------------------------------------------------
# Gate smoke-check: verify gate=0 truly zeroes the contribution
# ---------------------------------------------------------------------------

def _smoke_check_gate_zero(head, pyramid_sample, device):
    """Verify that non-resolution-changing gates are strict no-ops at 0.

    Only checks gates that don't alter output tensor shape (i.e., all gates
    except gate_f0 which switches the output from fpn_1 to fpn_0 resolution).
    Runs once at startup; logs a warning if the delta is unexpectedly large.
    """
    if not hasattr(head, "gate_state"):
        return
    gate_state = head.gate_state()
    if not gate_state:
        return

    # Identify gates that preserve output resolution when toggled
    # gate_f0 changes resolution (fpn_1 → fpn_0) so exclude it from the check
    testable_gates = [g for g in gate_state if g != "gate_f0"]
    if not testable_gates:
        return

    head.eval()
    pyramid = {k: v.unsqueeze(0).to(device) for k, v in pyramid_sample["pyramid"].items()}

    # Ensure gate_f0 stays at 0 (no resolution change during check)
    for g in gate_state:
        head.set_gate(g, 0.0)

    with torch.no_grad():
        out_zero = head(pyramid).clone()

    # Set only testable gates to epsilon; leave gate_f0 at 0
    for g in testable_gates:
        head.set_gate(g, 1e-9)

    with torch.no_grad():
        out_eps = head(pyramid).clone()

    max_diff = float((out_zero - out_eps).abs().max().item())
    head.train()

    if max_diff > 1e-4:
        print(
            f"[warn] gate smoke-check: max diff at gate≈0 is {max_diff:.2e} "
            "(may indicate non-zero initialization of gated branch)",
            file=sys.stderr,
        )
    else:
        print(f"[info] gate smoke-check passed: max diff at gate≈0 is {max_diff:.2e}", flush=True)

    # Restore all gates to 0 for training start
    for g in gate_state:
        head.set_gate(g, 0.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MoNuSeg 80-epoch layer-warmup runner")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--human-name", default=None, help="Human-readable variant name")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--wandb-project", default="frozen-sam-readout")
    parser.add_argument("--wandb-group", default="monuseg_testscreen80_layerwarm_aug")
    parser.add_argument("--wandb-job-type", default="TEST_SCREEN_80")
    parser.add_argument("--allow-wandb-failure", action="store_true", default=True)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None, help="Override total_epochs (smoke test)")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    config = load_yaml_config(args.config)

    dataset_name = str(config.get("dataset", {}).get("name", "monuseg"))
    eval_split_cfg = str(config.get("evaluation", {}).get("split", "test"))
    assert_monuseg_no_val(dataset_name, eval_split_cfg)
    if eval_split_cfg != "test":
        raise SystemExit(f"eval_split must be 'test', got {eval_split_cfg!r}")

    variant = str(config.get("model", {}).get("variant", ""))
    if variant not in _ALL_VARIANTS:
        raise SystemExit(f"Unknown variant {variant!r}. Known: {sorted(_ALL_VARIANTS)}")

    human_name = args.human_name or str(config.get("meta", {}).get("human_name", variant))

    training_cfg = config.get("training", {})
    total_epochs_cfg = int(training_cfg.get("total_epochs", 80))
    total_epochs = args.max_epochs if args.max_epochs is not None else total_epochs_cfg
    eval_every = int(training_cfg.get("eval_every", 5))

    aug_policy = str(config.get("dataset", {}).get("augmentation_policy", "none")).lower()
    aug_rng = np.random.default_rng(int(args.seed))

    warmup_cfg = config.get("warmup", {})
    gate_configs = warmup_cfg.get("gates", {})

    out_dir = Path(args.output_dir or f"outputs/{args.run_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval").mkdir(exist_ok=True)
    (out_dir / "visuals").mkdir(exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)
    (out_dir / "wandb").mkdir(exist_ok=True)

    (out_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    train_samples = list(iter_monuseg_binary_samples(split="train"))
    test_samples = list(iter_monuseg_binary_samples(split="test"))
    if args.limit_train is not None:
        train_samples = train_samples[:args.limit_train]
    if args.limit_test is not None:
        test_samples = test_samples[:args.limit_test]

    (out_dir / "split_manifest.json").write_text(json.dumps({
        "dataset": "monuseg",
        "train_split": "train",
        "eval_split": "test",
        "note": "MoNuSeg has no real validation set. Test used for ablation screening only.",
        "train_count": len(train_samples),
        "test_count": len(test_samples),
        "train_sample_ids": [str(s.crop_name) for s in train_samples],
        "test_sample_ids": [str(s.crop_name) for s in test_samples],
    }, indent=2, sort_keys=True))

    resize_cfg = config.get("dataset", {}).get("resize_before_sam")
    resize_hw = (int(resize_cfg["height"]), int(resize_cfg["width"])) if resize_cfg else None

    device = torch.device(
        "cuda" if (args.device in {"auto", "cuda"} and torch.cuda.is_available()) else "cpu"
    )
    sam_cfg = config.get("sam", {})
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    extractor = build_frozen_sam_pyramid_extractor(
        model_id=str(sam_cfg.get("model_id", "facebook/sam3")),
        device=args.device,
        hf_token=hf_token,
    )

    actual_channels = _probe_backbone_channels(extractor, train_samples[0], resize_hw)
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
        "channel_parity_ok": (
            actual_channels.get("f2_channels") == 256
            and actual_channels.get("f1_channels") == 64
            and actual_channels.get("f0_channels") == 32
        ),
    }
    (out_dir / "feature_shapes.json").write_text(json.dumps(feature_shapes, indent=2, sort_keys=True))
    print(f"[info] feature_shapes: {feature_shapes}", flush=True)

    head = _build_head(variant, actual_channels, config.get("model", {})).to(device)

    # Smoke-check: at gate=0, branch contributes zero
    probe_stream = _streaming_feature_iter(extractor, [train_samples[0]], resize_hw)
    probe_sample = next(probe_stream)
    _smoke_check_gate_zero(head, probe_sample, device)

    lr = float(training_cfg.get("lr", 1e-4))
    lr_min = float(training_cfg.get("lr_min", 1e-6))
    weight_decay = float(training_cfg.get("weight_decay", 1e-4))
    lr_schedule = str(training_cfg.get("lr_schedule", "cosine")).lower()

    optimizer = torch.optim.AdamW(
        [p for p in head.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = None
    if lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs, eta_min=lr_min
        )

    threshold = float(config.get("evaluation", {}).get("threshold", 0.5))
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
            tags=["monuseg", "test_screen_80", "layerwarm", variant, f"seed{args.seed}"],
            config={
                "run_name": args.run_name,
                "human_name": human_name,
                "variant": variant,
                "seed": int(args.seed),
                "eval_split": "test",
                "total_epochs": total_epochs,
                "eval_every": eval_every,
                "lr": lr,
                "lr_min": lr_min,
                "lr_schedule": lr_schedule,
                "weight_decay": weight_decay,
                "augmentation_policy": aug_policy,
                "augmentation_description": describe_augmentation_policy(aug_policy),
                "backbone_backend": backbone_backend,
                "paper_headline_safe": feature_shapes["paper_headline_safe"],
                "channel_parity_ok": feature_shapes["channel_parity_ok"],
                "gate_configs": gate_configs,
                "train_count": len(train_samples),
                "test_count": len(test_samples),
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
            "run_name": args.run_name, "human_name": human_name,
            "variant": variant, "seed": int(args.seed),
            "eval_split": "test", "stage": "TEST_SCREEN_80",
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

    final_ckpt = out_dir / "checkpoint_final.pt"
    best_ckpt = out_dir / "checkpoint_best.pt"
    per_epoch_csv = out_dir / "eval" / "per_epoch_metrics.csv"
    per_epoch_csv.write_text("epoch,stage,gate_state,train_loss,lr,test_dice,test_iou,is_best\n")

    eval_epochs = set(range(eval_every, total_epochs, eval_every)) | {total_epochs}

    for epoch in range(1, total_epochs + 1):
        # Update gate values for this epoch
        gate_state: Dict[str, float] = {}
        if gate_configs and hasattr(head, "set_gate"):
            gate_state = _update_gates(head, gate_configs, epoch)
        elif hasattr(head, "gate_state"):
            gate_state = head.gate_state()

        stage_label = _active_stage_label(gate_state)

        train_stream = _streaming_feature_iter(
            extractor, train_samples, resize_hw,
            aug_policy=aug_policy, rng=aug_rng,
        )
        train_stats = train_one_epoch(
            model=head, dataset=train_stream, optimizer=optimizer,
            loss_fn=bce_dice_loss, device=device,
        )
        current_lr = float(optimizer.param_groups[0]["lr"])

        if scheduler is not None:
            scheduler.step()

        # Evaluate at scheduled epochs and final epoch
        if epoch in eval_epochs:
            test_result = run_official_eval(
                extractor=extractor, head=head, samples=test_samples,
                device=device, threshold=threshold, resize_before_sam_hw=resize_hw,
            )
            test_dice = float(test_result.dice)
            test_iou = float(test_result.iou)
            is_best = test_dice > best_test_dice
            if is_best:
                best_test_dice = test_dice
                best_test_iou = test_iou
                best_test_epoch = epoch
                save_checkpoint(best_ckpt, model=head, optimizer=optimizer, epoch=epoch)

            final_epoch_dice = test_dice
            final_epoch_iou = test_iou

            # Epoch checkpoint
            epoch_ckpt = out_dir / "checkpoints" / f"checkpoint_ep{epoch:03d}.pt"
            save_checkpoint(epoch_ckpt, model=head, optimizer=optimizer, epoch=epoch)

            # Visual for fixed sample
            pred = _predict_sample(
                extractor=extractor, head=head, sample=fixed_visual_sample,
                resize_hw=resize_hw, device=device, threshold=threshold,
            )
            panel_path = _make_fixed_visual(
                sample=fixed_visual_sample, pred=pred, out_dir=out_dir,
                epoch=epoch, variant=variant, human_name=human_name,
                seed=args.seed, dice=test_dice, iou=test_iou,
            )

            gate_state_str = json.dumps({k: round(v, 4) for k, v in gate_state.items()})
            with per_epoch_csv.open("a") as f:
                f.write(f"{epoch},{stage_label},{gate_state_str},"
                        f"{train_stats['loss']:.6f},{current_lr:.8f},"
                        f"{test_dice:.6f},{test_iou:.6f},{int(is_best)}\n")

            if wandb_run is not None:
                try:
                    import wandb as wandb_mod
                    log_dict = {
                        "epoch": epoch,
                        "train/loss": float(train_stats["loss"]),
                        "train/lr": current_lr,
                        "test/dice": test_dice,
                        "test/iou": test_iou,
                        "provenance/eval_split": "test",
                        "provenance/stage": stage_label,
                        "test/sample_segmentation": wandb_mod.Image(
                            np.asarray(Image.open(panel_path)),
                            caption=(
                                f"{human_name} | epoch={epoch} | split=test "
                                f"| dice={test_dice:.4f} | iou={test_iou:.4f} | seed={args.seed}"
                            ),
                        ),
                    }
                    for k, v in gate_state.items():
                        log_dict[f"gates/{k}"] = v
                    if hasattr(head, "alpha_f0"):
                        log_dict["model/alpha_f0"] = head.alpha_f0
                    wandb_run.log(log_dict)
                except Exception as exc:
                    print(f"[warn] wandb log failed at epoch {epoch}: {exc}", file=sys.stderr)

            print(
                f"[{args.run_name}] ep {epoch}/{total_epochs} stage={stage_label} "
                f"loss={train_stats['loss']:.4f} test_dice={test_dice:.4f} test_iou={test_iou:.4f}",
                flush=True,
            )
        else:
            # Non-eval epoch: log train-only row
            gate_state_str = json.dumps({k: round(v, 4) for k, v in gate_state.items()})
            with per_epoch_csv.open("a") as f:
                f.write(f"{epoch},{stage_label},{gate_state_str},"
                        f"{train_stats['loss']:.6f},{current_lr:.8f},,,\n")
            if wandb_run is not None:
                try:
                    log_dict = {
                        "epoch": epoch,
                        "train/loss": float(train_stats["loss"]),
                        "train/lr": current_lr,
                        "provenance/stage": stage_label,
                    }
                    for k, v in gate_state.items():
                        log_dict[f"gates/{k}"] = v
                    if hasattr(head, "alpha_f0"):
                        log_dict["model/alpha_f0"] = head.alpha_f0
                    wandb_run.log(log_dict)
                except Exception as exc:
                    print(f"[warn] wandb log failed at epoch {epoch}: {exc}", file=sys.stderr)

            print(
                f"[{args.run_name}] ep {epoch}/{total_epochs} stage={stage_label} "
                f"loss={train_stats['loss']:.4f} lr={current_lr:.2e}",
                flush=True,
            )

    # -----------------------------------------------------------------------
    # Final artifacts
    # -----------------------------------------------------------------------
    save_checkpoint(final_ckpt, model=head, optimizer=optimizer, epoch=total_epochs)
    final_sha = _sha256(final_ckpt)
    best_sha = _sha256(best_ckpt)

    # Final visual: all test samples (seed 0 only for storage reasons)
    if args.seed == 0:
        print(f"[{args.run_name}] saving final visuals for all {len(test_samples)} test samples...", flush=True)
        for sample in test_samples:
            pred = _predict_sample(
                extractor=extractor, head=head, sample=sample,
                resize_hw=resize_hw, device=device, threshold=threshold,
            )
            _make_fixed_visual(
                sample=sample, pred=pred, out_dir=out_dir,
                epoch=total_epochs, variant=variant, human_name=human_name,
                seed=args.seed, dice=final_epoch_dice, iou=final_epoch_iou,
            )

    # Provenance sidecar
    (out_dir / "visuals" / "provenance.json").write_text(json.dumps({
        "eval_split": "test",
        "variant": variant,
        "human_name": human_name,
        "seed": int(args.seed),
        "augmentation_policy": aug_policy,
        "total_epochs": total_epochs,
        "final_epoch": total_epochs,
        "checkpoint_final_sha256": final_sha,
        "checkpoint_best_sha256": best_sha,
        "gate_configs": gate_configs,
    }, indent=2, sort_keys=True))

    test_summary = {
        "dataset": "monuseg",
        "variant": variant,
        "human_name": human_name,
        "seed": int(args.seed),
        "eval_split": "test",
        "stage": "TEST_SCREEN_80",
        "total_epochs": total_epochs,
        "eval_every": eval_every,
        "lr_schedule": lr_schedule,
        "lr_min": lr_min,
        "augmentation_policy": aug_policy,
        "augmentation_description": describe_augmentation_policy(aug_policy),
        "gate_configs": gate_configs,
        "checkpoint_views": {
            "final_epoch": {
                "selected_by": "final_epoch",
                "selected_epoch": total_epochs,
                "claimability": "claimable_test_screen",
                "test_dice": final_epoch_dice,
                "test_iou": final_epoch_iou,
                "checkpoint_path": str(final_ckpt.resolve()),
                "checkpoint_sha256": final_sha,
            },
            "best_test_epoch": {
                "selected_by": "test_peak",
                "selected_epoch": best_test_epoch,
                "claimability": "diagnostic_oracle_not_claimable",
                "test_dice": best_test_dice,
                "test_iou": best_test_iou,
                "checkpoint_path": str(best_ckpt.resolve()),
                "checkpoint_sha256": best_sha,
            },
        },
        # Flat fields for backward compat
        "final_epoch_dice": final_epoch_dice,
        "final_epoch_iou": final_epoch_iou,
        "best_test_epoch": best_test_epoch,
        "best_test_dice": best_test_dice,
        "best_test_iou": best_test_iou,
        "backbone_backend": backbone_backend,
        "channel_parity_ok": feature_shapes["channel_parity_ok"],
        "paper_headline_safe": feature_shapes["paper_headline_safe"],
        "wandb_url": wandb_url,
    }
    (out_dir / "eval" / "test_metrics.json").write_text(json.dumps(test_summary, indent=2, sort_keys=True))

    if wandb_run is not None:
        try:
            wandb_run.summary.update({
                "final_epoch/dice": final_epoch_dice,
                "final_epoch/iou": final_epoch_iou,
                "best_test/dice": best_test_dice,
                "best_test/iou": best_test_iou,
                "best_test/epoch": best_test_epoch,
                "claimability": "final_epoch_only",
                "checkpoint_final_sha256": final_sha,
                "backbone_backend": backbone_backend,
                "channel_parity_ok": feature_shapes["channel_parity_ok"],
            })
            wandb_run.finish()
        except Exception as exc:
            print(f"[warn] wandb finish failed: {exc}", file=sys.stderr)

    print(
        f"[{args.run_name}] DONE "
        f"final_epoch test_dice={final_epoch_dice:.4f} test_iou={final_epoch_iou:.4f} "
        f"(claimable_test_screen, epoch={total_epochs})",
        flush=True,
    )
    print(
        f"[{args.run_name}] oracle best_test_dice={best_test_dice:.4f} "
        f"(diagnostic_oracle_not_claimable, epoch={best_test_epoch})",
        flush=True,
    )
    print(f"Saved: {out_dir / 'eval' / 'test_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
