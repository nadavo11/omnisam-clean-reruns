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
from frozen_sam_readout.evaluation.official_eval import run_official_eval, write_per_sample_csv
from frozen_sam_readout.evaluation.metrics import compute_binary_metrics
from frozen_sam_readout.models.registry import build_head
from frozen_sam_readout.sam import LearnedPromptedSam3, build_frozen_sam_pyramid_extractor
from frozen_sam_readout.training import bce_dice_loss, load_checkpoint, save_checkpoint, train_one_epoch
from frozen_sam_readout.training.augmentations import apply_augmentation, describe_augmentation_policy
from frozen_sam_readout.utils import load_yaml_config, set_seed, write_run_manifest
from frozen_sam_readout.utils.config import assert_monuseg_no_val
from frozen_sam_readout.visualization import compose_strip, binary_error_map, render_gt_overlay, render_single_mask_overlay

_SINGLE_STAGE_VARIANTS = {
    "a0_fpn2",
    "a2_fpn2_fpn1_refine",
    "a3_memory",
    "d0_linear",
    "d0_small_head",
    "fpn_plus_d0",
    "d0_fpn_fusion",
    "d0_f0_fusion",
    "d0_f1_fusion",
    "d0_f2_fusion",
    "d0_f0_f1_fusion",
    "d0_f1_f2_fusion",
    "d0_f0_f2_fusion",
    "d0fpn_source_gated_se",
    "d0fpn_cbam",
    "d0_query_fpn_crossattn",
    "fpn_query_d0_crossattn",
    "d0fpn_bidirectional_crossattn",
    "d0fpn_task_tokens_film",
    "d0fpn_window_selfattn",
    "d0fpn_global_context_nonlocal",
    "d0fpn_task_tokens_film_spatial_after",
    "d0fpn_spatial_before_task_tokens_film",
    "d0fpn_task_conditioned_spatial_attention",
    "d0fpn_residual_task_spatial_delta",
    "d0fpn_logit_residual_task_spatial",
    "d0fpn_dual_skip_task_spatial_fusion",
    "d0fpn_source_spatial_gates_task_tokens",
    "d0fpn_f0_boundary_spatial_task_refine",
    "d0fpn_cap160_task_tokens_16_film_wide",
    "d0fpn_cap160_task_tokens_32_film_wide",
    "d0fpn_cap160_task_spatial_residual_2layer",
    "d0fpn_cap160_task_spatial_residual_4layer",
    "d0fpn_cap160_fine_map_fusion_highres_head",
    "d0fpn_cap160_wide_fusion_dim256",
    "d0fpn_cap160_deep_conv_fusion_residual",
    "d0fpn_cap160_hybrid_best_task_spatial_fine",
}
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
    feature_control: dict | None = None,
    seed: int = 0,
) -> Iterator[dict]:
    for sample in samples:
        if aug_policy != "none" and rng is not None:
            # Augmentation applied to PIL image + mask BEFORE SAM feature extraction
            sample = apply_augmentation(sample, rng=rng, policy=aug_policy)
        img = _resize_image(sample.image, resize_hw)
        _, pyramid_np = extractor.extract_sam_pyramid(img)
        pyramid_np = _apply_decoder_map_control(
            pyramid_np,
            sample_id=str(sample.crop_name),
            seed=seed,
            control=feature_control,
        )
        pyramid_t = {
            name: torch.from_numpy(np.ascontiguousarray(f)).float()
            for name, f in pyramid_np.items()
        }
        target = torch.from_numpy(np.asarray(sample.texture_a_mask, dtype=np.float32))
        yield {"pyramid": pyramid_t, "target": target}


def _prompt_cfg(config: dict) -> dict:
    return dict(config.get("prompt", {}))


def _is_learned_prompt_mode(config: dict) -> bool:
    mode = str(_prompt_cfg(config).get("mode", "fixed_full_image_box"))
    return mode in {
        "learned_constant_sparse",
        "fixed_full_image_box_plus_learned_delta",
        "fixed_full_image_box_plus_learned_text_soft_prompt",
    }


def _train_one_epoch_prompted(
    *,
    prompt_model: LearnedPromptedSam3,
    head: torch.nn.Module,
    samples: Iterable,
    resize_hw,
    aug_policy: str,
    rng: "np.random.Generator | None",
    optimizer: torch.optim.Optimizer,
    loss_fn,
    device: torch.device,
) -> dict[str, float]:
    head.train()
    prompt_model.train()
    total_loss = 0.0
    count = 0
    for sample in samples:
        if aug_policy != "none" and rng is not None:
            sample = apply_augmentation(sample, rng=rng, policy=aug_policy)
        image = _resize_image(sample.image, resize_hw)
        pyramid = prompt_model.forward_pyramid(image, capture_numpy=False)
        logits = head({k: v.unsqueeze(0).to(device) for k, v in pyramid.items()})
        target = torch.from_numpy(np.asarray(sample.texture_a_mask, dtype=np.float32)).to(device)
        target_resized = F.interpolate(
            target[None, None, ...], size=logits.shape[-2:], mode="bilinear", align_corners=False
        )
        loss = loss_fn(logits, target_resized)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        count += 1
    return {"loss": total_loss / max(count, 1)}


def _predict_sample(*, extractor, head, sample, resize_hw, device, threshold, feature_control=None, seed: int = 0):
    image = _resize_image(sample.image, resize_hw)
    _, pyramid_np = extractor.extract_sam_pyramid(image)
    pyramid_np = _apply_decoder_map_control(
        pyramid_np,
        sample_id=str(sample.crop_name),
        seed=seed,
        control=feature_control,
    )
    tensors = {
        k: torch.from_numpy(np.ascontiguousarray(v)).float().unsqueeze(0).to(device)
        for k, v in pyramid_np.items()
    }
    with torch.inference_mode():
        logits = head(tensors)
        logits_up = F.interpolate(logits, size=sample.texture_a_mask.shape, mode="bilinear", align_corners=False)
        return (torch.sigmoid(logits_up)[0, 0].cpu().numpy() > threshold).astype(bool)


def _stable_rng(*, sample_id: str, seed: int, purpose: str) -> np.random.Generator:
    payload = f"{purpose}|{sample_id}|{int(seed)}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)
    return np.random.default_rng(value)


def _apply_decoder_map_control(
    pyramid_np: dict[str, np.ndarray],
    *,
    sample_id: str,
    seed: int,
    control: dict | None,
) -> dict[str, np.ndarray]:
    if not control or str(control.get("mode", "none")).lower() == "none":
        return pyramid_np
    if "decoder_semantic_map" not in pyramid_np:
        return pyramid_np
    mode = str(control.get("mode", "none")).lower()
    d0 = np.asarray(pyramid_np["decoder_semantic_map"], dtype=np.float32)
    c, h, w = d0.shape
    rng = _stable_rng(sample_id=sample_id, seed=seed, purpose=mode)
    out = dict(pyramid_np)
    if mode == "spatial_permutation":
        perm = rng.permutation(h * w)
        out["decoder_semantic_map"] = np.ascontiguousarray(d0.reshape(c, h * w)[:, perm].reshape(c, h, w))
    elif mode == "block_shuffle":
        block = int(control.get("block_size", 8))
        if h % block != 0 or w % block != 0:
            raise ValueError(f"block_shuffle requires D0 H/W divisible by block_size={block}, got {(h, w)}")
        bh, bw = h // block, w // block
        blocks = d0.reshape(c, bh, block, bw, block).transpose(1, 3, 0, 2, 4).reshape(bh * bw, c, block, block)
        perm = rng.permutation(bh * bw)
        shuffled = blocks[perm].reshape(bh, bw, c, block, block).transpose(2, 0, 3, 1, 4).reshape(c, h, w)
        out["decoder_semantic_map"] = np.ascontiguousarray(shuffled)
    elif mode == "channel_shuffle":
        perm = rng.permutation(c)
        out["decoder_semantic_map"] = np.ascontiguousarray(d0[perm])
    elif mode == "channel_sign_shuffle":
        perm = rng.permutation(c)
        signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=(c, 1, 1))
        out["decoder_semantic_map"] = np.ascontiguousarray(d0[perm] * signs)
    else:
        raise ValueError(f"Unsupported decoder_map_control.mode={mode!r}")
    return out


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


def _write_final_visual_package(*, extractor, head, samples, resize_hw, device, threshold, out_dir, variant, seed, feature_control=None) -> Path:
    package_dir = out_dir / "visuals" / "final_epoch_all_test"
    package_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for sample in samples:
        pred = _predict_sample(
            extractor=extractor,
            head=head,
            sample=sample,
            resize_hw=resize_hw,
            device=device,
            threshold=threshold,
            feature_control=feature_control,
            seed=seed,
        )
        image = sample.image.convert("RGB")
        gt = np.asarray(sample.texture_a_mask, dtype=bool)
        sample_dir = package_dir / str(sample.crop_name)
        sample_dir.mkdir(parents=True, exist_ok=True)
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
        metrics = compute_binary_metrics(pred, gt)
        manifest.append({
            "sample_id": str(sample.crop_name),
            "eval_split": "test",
            "dice": float(metrics.dice),
            "iou": float(metrics.iou),
            "pred_area": int(pred.sum()),
            "gt_area": int(gt.sum()),
            "panel": str((sample_dir / "panel.png").resolve()),
        })
    (package_dir / "manifest.json").write_text(json.dumps({
        "variant": variant,
        "seed": int(seed),
        "eval_split": "test",
        "count": len(manifest),
        "samples": manifest,
    }, indent=2, sort_keys=True))
    return package_dir


def _probe_backbone_channels(extractor, sample, resize_hw):
    img = _resize_image(sample.image, resize_hw)
    _, pyramid_np = extractor.extract_sam_pyramid(img)
    defaults = {"f2_channels": 256, "f1_channels": 64, "f0_channels": 32}
    for level, key in [("fpn_2", "f2_channels"), ("fpn_1", "f1_channels"), ("fpn_0", "f0_channels")]:
        if level in pyramid_np:
            defaults[key] = int(pyramid_np[level].shape[0])
    if "decoder_semantic_map" in pyramid_np:
        defaults["decoder_map_channels"] = int(pyramid_np["decoder_semantic_map"].shape[0])
    return defaults


def _probe_feature_map_shapes(extractor, sample, resize_hw) -> dict[str, list[int]]:
    img = _resize_image(sample.image, resize_hw)
    _, pyramid_np = extractor.extract_sam_pyramid(img)
    return {name: [int(v) for v in fmap.shape] for name, fmap in pyramid_np.items()}


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
    elif variant == "d0_linear":
        kwargs = {"decoder_map_channels": int(channels["decoder_map_channels"])}
    elif variant == "d0_small_head":
        kwargs = {"decoder_map_channels": int(channels["decoder_map_channels"])}
    elif variant == "fpn_plus_d0":
        kwargs.pop("f0_channels", None)
        kwargs["decoder_map_channels"] = int(channels["decoder_map_channels"])
        kwargs["memory_tokens"] = int(model_cfg.get("memory_tokens", 4))
    elif variant == "d0_fpn_fusion":
        kwargs["decoder_map_channels"] = int(channels["decoder_map_channels"])
        kwargs["projection_dim"] = int(model_cfg.get("projection_dim", 64))
    elif variant == "d0_f0_fusion":
        kwargs = {
            "f0_channels": int(channels["f0_channels"]),
            "decoder_map_channels": int(channels["decoder_map_channels"]),
            "projection_dim": int(model_cfg.get("projection_dim", 64)),
        }
    elif variant in {"d0_f1_fusion", "d0_f2_fusion", "d0_f0_f1_fusion", "d0_f1_f2_fusion", "d0_f0_f2_fusion"}:
        source_key_map = {
            "d0_f1_fusion": ("fpn_1",),
            "d0_f2_fusion": ("fpn_2",),
            "d0_f0_f1_fusion": ("fpn_0", "fpn_1"),
            "d0_f1_f2_fusion": ("fpn_1", "fpn_2"),
            "d0_f0_f2_fusion": ("fpn_0", "fpn_2"),
        }
        channel_key_map = {
            "fpn_0": "f0_channels",
            "fpn_1": "f1_channels",
            "fpn_2": "f2_channels",
        }
        source_keys = source_key_map[variant]
        source_channels = {"decoder_semantic_map": int(channels["decoder_map_channels"])}
        for key in source_keys:
            source_channels[key] = int(channels[channel_key_map[key]])
        kwargs = {
            "source_channels": source_channels,
            "source_keys": ("decoder_semantic_map", *source_keys),
            "projection_dim": int(model_cfg.get("projection_dim", 64)),
        }
    elif variant in {
        "d0fpn_source_gated_se",
        "d0fpn_cbam",
        "d0_query_fpn_crossattn",
        "fpn_query_d0_crossattn",
        "d0fpn_bidirectional_crossattn",
        "d0fpn_task_tokens_film",
        "d0fpn_window_selfattn",
        "d0fpn_global_context_nonlocal",
        "d0fpn_task_tokens_film_spatial_after",
        "d0fpn_spatial_before_task_tokens_film",
        "d0fpn_task_conditioned_spatial_attention",
        "d0fpn_residual_task_spatial_delta",
        "d0fpn_logit_residual_task_spatial",
        "d0fpn_dual_skip_task_spatial_fusion",
        "d0fpn_source_spatial_gates_task_tokens",
        "d0fpn_f0_boundary_spatial_task_refine",
    }:
        source_channels = {
            "decoder_semantic_map": int(channels["decoder_map_channels"]),
            "fpn_2": int(channels["f2_channels"]),
            "fpn_1": int(channels["f1_channels"]),
            "fpn_0": int(channels["f0_channels"]),
        }
        kwargs = {
            "source_channels": source_channels,
            "source_keys": ("decoder_semantic_map", "fpn_2", "fpn_1", "fpn_0"),
            "projection_dim": int(model_cfg.get("projection_dim", 64)),
            "decoder_dim": int(model_cfg.get("decoder_dim", 128)),
        }
        if variant == "d0fpn_source_gated_se":
            kwargs["gate_scale"] = float(model_cfg.get("gate_scale", 0.1))
        elif variant == "d0fpn_cbam":
            kwargs["reduction"] = int(model_cfg.get("cbam_reduction", 4))
        elif variant in {"d0_query_fpn_crossattn", "fpn_query_d0_crossattn", "d0fpn_bidirectional_crossattn"}:
            kwargs["num_heads"] = int(model_cfg.get("num_heads", 4))
            kwargs["pool_hw"] = int(model_cfg.get("pool_hw", 16))
        elif variant == "d0fpn_task_tokens_film":
            kwargs["num_tokens"] = int(model_cfg.get("num_tokens", 4))
            kwargs["num_heads"] = int(model_cfg.get("num_heads", 4))
            kwargs["pool_hw"] = int(model_cfg.get("pool_hw", 16))
        elif variant == "d0fpn_window_selfattn":
            kwargs["num_heads"] = int(model_cfg.get("num_heads", 4))
            kwargs["pool_hw"] = int(model_cfg.get("pool_hw", 16))
            kwargs["window_size"] = int(model_cfg.get("window_size", 4))
        elif variant == "d0fpn_global_context_nonlocal":
            kwargs["num_heads"] = int(model_cfg.get("num_heads", 4))
            kwargs["query_hw"] = int(model_cfg.get("query_hw", 16))
            kwargs["context_hw"] = int(model_cfg.get("context_hw", 8))
        elif variant in {
            "d0fpn_task_tokens_film_spatial_after",
            "d0fpn_spatial_before_task_tokens_film",
            "d0fpn_task_conditioned_spatial_attention",
            "d0fpn_residual_task_spatial_delta",
            "d0fpn_logit_residual_task_spatial",
            "d0fpn_dual_skip_task_spatial_fusion",
            "d0fpn_source_spatial_gates_task_tokens",
            "d0fpn_f0_boundary_spatial_task_refine",
        }:
            kwargs["num_tokens"] = int(model_cfg.get("num_tokens", 4))
            kwargs["num_heads"] = int(model_cfg.get("num_heads", 4))
            kwargs["pool_hw"] = int(model_cfg.get("pool_hw", 16))
            kwargs["alpha_init"] = float(model_cfg.get("alpha_init", 0.05))
    elif variant in {
        "d0fpn_cap160_task_tokens_16_film_wide",
        "d0fpn_cap160_task_tokens_32_film_wide",
        "d0fpn_cap160_task_spatial_residual_2layer",
        "d0fpn_cap160_task_spatial_residual_4layer",
        "d0fpn_cap160_fine_map_fusion_highres_head",
        "d0fpn_cap160_wide_fusion_dim256",
        "d0fpn_cap160_deep_conv_fusion_residual",
        "d0fpn_cap160_hybrid_best_task_spatial_fine",
    }:
        source_channels = {
            "decoder_semantic_map": int(channels["decoder_map_channels"]),
            "fpn_2": int(channels["f2_channels"]),
            "fpn_1": int(channels["f1_channels"]),
            "fpn_0": int(channels["f0_channels"]),
        }
        kwargs = {
            "source_channels": source_channels,
            "source_keys": ("decoder_semantic_map", "fpn_2", "fpn_1", "fpn_0"),
            "projection_dim": int(model_cfg.get("projection_dim", 96)),
            "decoder_dim": int(model_cfg.get("decoder_dim", 192)),
        }
        if variant == "d0fpn_cap160_task_tokens_16_film_wide":
            kwargs["num_tokens"] = int(model_cfg.get("num_tokens", 16))
            kwargs["num_heads"] = int(model_cfg.get("num_heads", 8))
            kwargs["pool_hw"] = int(model_cfg.get("pool_hw", 16))
            kwargs["token_hidden_dim"] = int(model_cfg.get("token_hidden_dim", 512))
            kwargs["warmup_epochs"] = int(model_cfg.get("warmup_epochs", 20))
        elif variant == "d0fpn_cap160_task_tokens_32_film_wide":
            kwargs["num_tokens"] = int(model_cfg.get("num_tokens", 32))
            kwargs["num_heads"] = int(model_cfg.get("num_heads", 8))
            kwargs["pool_hw"] = int(model_cfg.get("pool_hw", 16))
            kwargs["token_hidden_dim"] = int(model_cfg.get("token_hidden_dim", 768))
            kwargs["warmup_epochs"] = int(model_cfg.get("warmup_epochs", 20))
        elif variant in {"d0fpn_cap160_task_spatial_residual_2layer", "d0fpn_cap160_task_spatial_residual_4layer"}:
            kwargs["num_tokens"] = int(model_cfg.get("num_tokens", 8))
            kwargs["num_heads"] = int(model_cfg.get("num_heads", 8))
            kwargs["pool_hw"] = int(model_cfg.get("pool_hw", 16))
            kwargs["num_blocks"] = int(model_cfg.get("num_blocks", 2 if variant.endswith("2layer") else 4))
            kwargs["refine_depth"] = int(model_cfg.get("refine_depth", 2))
            kwargs["warmup_epochs"] = int(model_cfg.get("warmup_epochs", 20))
        elif variant == "d0fpn_cap160_fine_map_fusion_highres_head":
            kwargs["warmup_epochs"] = int(model_cfg.get("warmup_epochs", 20))
        elif variant == "d0fpn_cap160_wide_fusion_dim256":
            # Same architecture family as the locked D0+FPN fusion, but with a wider head.
            kwargs["projection_dim"] = int(model_cfg.get("projection_dim", 128))
            kwargs["decoder_dim"] = int(model_cfg.get("decoder_dim", 256))
        elif variant == "d0fpn_cap160_deep_conv_fusion_residual":
            kwargs["num_blocks"] = int(model_cfg.get("num_blocks", 8))
            kwargs["task_every"] = int(model_cfg.get("task_every", 2))
            kwargs["num_tokens"] = int(model_cfg.get("num_tokens", 8))
            kwargs["num_heads"] = int(model_cfg.get("num_heads", 8))
            kwargs["pool_hw"] = int(model_cfg.get("pool_hw", 16))
            kwargs["warmup_epochs"] = int(model_cfg.get("warmup_epochs", 20))
        elif variant == "d0fpn_cap160_hybrid_best_task_spatial_fine":
            kwargs["num_tokens"] = int(model_cfg.get("num_tokens", 16))
            kwargs["num_heads"] = int(model_cfg.get("num_heads", 8))
            kwargs["pool_hw"] = int(model_cfg.get("pool_hw", 16))
            kwargs["task_blocks"] = int(model_cfg.get("task_blocks", 2))
            kwargs["refine_depth"] = int(model_cfg.get("refine_depth", 2))
            kwargs["warmup_epochs"] = int(model_cfg.get("warmup_epochs", 20))
    kwargs.pop("decoder_dim", None)
    kwargs["decoder_dim"] = int(model_cfg.get("decoder_dim", 128))
    return build_head(variant, **kwargs)


def _head_param_count(head: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in head.parameters() if p.requires_grad))


def _feature_sources_for_variant(variant: str) -> list[str]:
    mapping = {
        "a0_fpn2": ["fpn_2"],
        "a2_fpn2_fpn1_refine": ["fpn_2", "fpn_1"],
        "a3_memory": ["fpn_2", "fpn_1"],
        "a5_all_at_once": ["fpn_2", "fpn_1", "fpn_0"],
        "final_staged": ["fpn_2", "fpn_1", "fpn_0"],
        "final_staged_a2": ["fpn_2", "fpn_1", "fpn_0"],
        "d0_linear": ["decoder_semantic_map"],
        "d0_small_head": ["decoder_semantic_map"],
        "fpn_plus_d0": ["decoder_semantic_map", "fpn_2", "fpn_1"],
        "d0_fpn_fusion": ["decoder_semantic_map", "fpn_2", "fpn_1", "fpn_0"],
        "d0_f0_fusion": ["decoder_semantic_map", "fpn_0"],
        "d0_f1_fusion": ["decoder_semantic_map", "fpn_1"],
        "d0_f2_fusion": ["decoder_semantic_map", "fpn_2"],
        "d0_f0_f1_fusion": ["decoder_semantic_map", "fpn_0", "fpn_1"],
        "d0_f1_f2_fusion": ["decoder_semantic_map", "fpn_1", "fpn_2"],
        "d0_f0_f2_fusion": ["decoder_semantic_map", "fpn_0", "fpn_2"],
        "d0fpn_source_gated_se": ["decoder_semantic_map", "fpn_2", "fpn_1", "fpn_0"],
        "d0fpn_cbam": ["decoder_semantic_map", "fpn_2", "fpn_1", "fpn_0"],
        "d0_query_fpn_crossattn": ["decoder_semantic_map", "fpn_2", "fpn_1", "fpn_0"],
        "fpn_query_d0_crossattn": ["decoder_semantic_map", "fpn_2", "fpn_1", "fpn_0"],
        "d0fpn_bidirectional_crossattn": ["decoder_semantic_map", "fpn_2", "fpn_1", "fpn_0"],
        "d0fpn_task_tokens_film": ["decoder_semantic_map", "fpn_2", "fpn_1", "fpn_0"],
        "d0fpn_window_selfattn": ["decoder_semantic_map", "fpn_2", "fpn_1", "fpn_0"],
        "d0fpn_global_context_nonlocal": ["decoder_semantic_map", "fpn_2", "fpn_1", "fpn_0"],
        "d0fpn_task_tokens_film_spatial_after": ["decoder_semantic_map", "fpn_2", "fpn_1", "fpn_0"],
        "d0fpn_spatial_before_task_tokens_film": ["decoder_semantic_map", "fpn_2", "fpn_1", "fpn_0"],
        "d0fpn_task_conditioned_spatial_attention": ["decoder_semantic_map", "fpn_2", "fpn_1", "fpn_0"],
        "d0fpn_residual_task_spatial_delta": ["decoder_semantic_map", "fpn_2", "fpn_1", "fpn_0"],
        "d0fpn_logit_residual_task_spatial": ["decoder_semantic_map", "fpn_2", "fpn_1", "fpn_0"],
        "d0fpn_dual_skip_task_spatial_fusion": ["decoder_semantic_map", "fpn_2", "fpn_1", "fpn_0"],
        "d0fpn_source_spatial_gates_task_tokens": ["decoder_semantic_map", "fpn_2", "fpn_1", "fpn_0"],
        "d0fpn_f0_boundary_spatial_task_refine": ["decoder_semantic_map", "fpn_2", "fpn_1", "fpn_0"],
        "d0fpn_cap160_task_tokens_16_film_wide": ["decoder_semantic_map", "fpn_2", "fpn_1", "fpn_0"],
        "d0fpn_cap160_task_tokens_32_film_wide": ["decoder_semantic_map", "fpn_2", "fpn_1", "fpn_0"],
        "d0fpn_cap160_task_spatial_residual_2layer": ["decoder_semantic_map", "fpn_2", "fpn_1", "fpn_0"],
        "d0fpn_cap160_task_spatial_residual_4layer": ["decoder_semantic_map", "fpn_2", "fpn_1", "fpn_0"],
        "d0fpn_cap160_fine_map_fusion_highres_head": ["decoder_semantic_map", "fpn_2", "fpn_1", "fpn_0"],
        "d0fpn_cap160_wide_fusion_dim256": ["decoder_semantic_map", "fpn_2", "fpn_1", "fpn_0"],
        "d0fpn_cap160_deep_conv_fusion_residual": ["decoder_semantic_map", "fpn_2", "fpn_1", "fpn_0"],
        "d0fpn_cap160_hybrid_best_task_spatial_fine": ["decoder_semantic_map", "fpn_2", "fpn_1", "fpn_0"],
    }
    return mapping.get(variant, [])


def _normalize_features_chw(feature: np.ndarray) -> np.ndarray:
    flat = feature.reshape(feature.shape[0], -1).T
    flat = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8)
    return flat.T.reshape(feature.shape)


def _cosine_similarity_map(feature_chw: np.ndarray, mask_hw: np.ndarray) -> np.ndarray:
    feat = _normalize_features_chw(feature_chw)
    flat = feat.reshape(feat.shape[0], -1).T
    mask = np.asarray(mask_hw, dtype=bool).reshape(-1)
    if not mask.any():
        return np.zeros(feature_chw.shape[-2:], dtype=np.float32)
    proto = flat[mask].mean(axis=0)
    proto = proto / (np.linalg.norm(proto) + 1e-8)
    return np.asarray(flat @ proto, dtype=np.float32).reshape(feature_chw.shape[-2:])


def _boundary_energy(feature_chw: np.ndarray) -> np.ndarray:
    d_y = np.linalg.norm(feature_chw[:, 1:, :] - feature_chw[:, :-1, :], axis=0)
    d_x = np.linalg.norm(feature_chw[:, :, 1:] - feature_chw[:, :, :-1], axis=0)
    out = np.zeros(feature_chw.shape[-2:], dtype=np.float32)
    out[:-1, :] += d_y
    out[:, :-1] += d_x
    return out


def _semantic_coherence_for_sample(*, decoder_map: np.ndarray, gt: np.ndarray, pred: np.ndarray, boundary: np.ndarray) -> dict:
    feat_t = torch.from_numpy(np.ascontiguousarray(decoder_map)).float().unsqueeze(0)
    feat_up = F.interpolate(feat_t, size=gt.shape, mode="bilinear", align_corners=False)[0].numpy()
    fg_sim = _cosine_similarity_map(feat_up, gt)
    bg_sim = _cosine_similarity_map(feat_up, ~gt)
    margin = fg_sim - bg_sim
    proto_pred = margin >= 0.0
    proto_metrics = compute_binary_metrics(proto_pred, gt)
    boundary_energy = _boundary_energy(feat_up)
    gt_boundary = np.asarray(boundary, dtype=bool)
    non_boundary = ~gt_boundary
    flat = _normalize_features_chw(feat_up).reshape(feat_up.shape[0], -1).T
    pred_flat = np.asarray(pred, dtype=bool).reshape(-1)
    pred_intra = float("nan")
    if int(pred_flat.sum()) > 1:
        pred_vectors = flat[pred_flat]
        pred_proto = pred_vectors.mean(axis=0)
        pred_proto = pred_proto / (np.linalg.norm(pred_proto) + 1e-8)
        pred_intra = float((pred_vectors @ pred_proto).mean())
    try:
        from sklearn.metrics import roc_auc_score

        proto_auc = float(roc_auc_score(gt.reshape(-1).astype(np.uint8), margin.reshape(-1)))
    except Exception:
        proto_auc = float("nan")
    return {
        "fg_bg_gap": float(fg_sim[gt].mean() - fg_sim[~gt].mean()),
        "proto_auc": proto_auc,
        "proto_margin_dice": float(proto_metrics.dice),
        "proto_margin_iou": float(proto_metrics.iou),
        "pred_intra_cosine": pred_intra,
        "boundary_contrast": float(boundary_energy[gt_boundary].mean() - boundary_energy[non_boundary].mean()),
    }


def _write_semantic_coherence_diagnostics(*, extractor, head, samples, resize_hw, device, threshold, out_dir: Path, feature_control=None, seed: int = 0) -> dict:
    rows = []
    for index, sample in enumerate(samples):
        image = _resize_image(sample.image, resize_hw)
        _, pyramid_np = extractor.extract_sam_pyramid(image)
        pyramid_np = _apply_decoder_map_control(
            pyramid_np,
            sample_id=str(sample.crop_name),
            seed=seed,
            control=feature_control,
        )
        decoder_map = pyramid_np.get("decoder_semantic_map")
        if decoder_map is None:
            return {}
        tensors = {
            k: torch.from_numpy(np.ascontiguousarray(v)).float().unsqueeze(0).to(device)
            for k, v in pyramid_np.items()
        }
        with torch.inference_mode():
            logits = head(tensors)
            logits_up = F.interpolate(logits, size=sample.texture_a_mask.shape, mode="bilinear", align_corners=False)
            pred = (torch.sigmoid(logits_up)[0, 0].detach().cpu().numpy() > threshold)
        gt = np.asarray(sample.texture_a_mask, dtype=bool)
        row = {
            "sample_index": index,
            "sample_id": str(sample.crop_name),
            "eval_split": "test",
            **_semantic_coherence_for_sample(
                decoder_map=decoder_map,
                gt=gt,
                pred=pred,
                boundary=np.asarray(sample.boundary_mask, dtype=bool),
            ),
        }
        rows.append(row)
    csv_path = out_dir / "eval" / "semantic_coherence.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w") as handle:
        handle.write(",".join(fieldnames) + "\n")
        for row in rows:
            handle.write(",".join(str(row[key]) for key in fieldnames) + "\n")
    means = {
        key: float(np.nanmean([float(row[key]) for row in rows]))
        for key in ["fg_bg_gap", "proto_auc", "proto_margin_dice", "proto_margin_iou", "pred_intra_cosine", "boundary_contrast"]
    }
    payload = {"eval_split": "test", "rows": len(rows), "csv": str(csv_path.resolve()), "mean": means}
    (out_dir / "eval" / "semantic_coherence_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


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
    parser.add_argument("--max-epochs", type=int, default=None)
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
    evaluation_cfg = config.get("evaluation", {})
    logging_cfg = config.get("logging", {})
    feature_control = config.get("feature_control", {})
    stage1_epochs = int(training_cfg.get("stage1_epochs", 40))
    stage2_epochs = int(training_cfg.get("stage2_epochs", 0))
    if args.max_epochs is not None:
        stage1_epochs = min(stage1_epochs, int(args.max_epochs))
        if stage1_epochs < int(training_cfg.get("stage1_epochs", 40)):
            stage2_epochs = 0
    total_epochs = stage1_epochs + stage2_epochs
    eval_every = max(1, int(evaluation_cfg.get("eval_every", 1)))
    visual_every = max(1, int(logging_cfg.get("visual_every", total_epochs)))
    log_visuals_to_wandb = bool(logging_cfg.get("log_visuals_to_wandb", True))
    is_two_stage = variant in _TWO_STAGE_VARIANTS and stage2_epochs > 0

    aug_policy = str(config.get("dataset", {}).get("augmentation_policy", "none")).lower()
    aug_rng = np.random.default_rng(int(args.seed))  # seeded separately from model init

    out_dir = Path(args.output_dir or f"outputs/{args.run_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval").mkdir(exist_ok=True)
    (out_dir / "visuals").mkdir(exist_ok=True)
    (out_dir / "wandb").mkdir(exist_ok=True)

    # Save config and run metadata. Keep config.yaml for legacy readers, and
    # resolved_config.yaml for the locked test-screen artifact contract.
    resolved_config_text = yaml.safe_dump(config, sort_keys=False)
    (out_dir / "config.yaml").write_text(resolved_config_text)
    (out_dir / "resolved_config.yaml").write_text(resolved_config_text)

    # Load data — full training set, no val holdout. For this semantic-map
    # batch, "full" means all 37 HF train rows, including the 7 tissue==0 rows.
    train_selection_policy = str(
        config.get("dataset", {}).get(
            "train_selection_policy",
            "official_challenge_train_excludes_tissue_0_unknown",
        )
    )
    include_extra_train_unknown = train_selection_policy in {
        "hf_train_all_37",
        "hf_train_all_37_includes_tissue_0_unknown",
        "full_37",
    }
    train_samples = list(
        iter_monuseg_binary_samples(
            split="train",
            include_extra_train_unknown=include_extra_train_unknown,
        )
    )
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
        "train_selection_policy": (
            "hf_train_all_37_includes_tissue_0_unknown"
            if include_extra_train_unknown
            else "official_challenge_train_excludes_tissue_0_unknown"
        ),
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
    prompt_cfg = _prompt_cfg(config)
    prompt_mode = str(prompt_cfg.get("mode", "fixed_full_image_box"))
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if _is_learned_prompt_mode(config):
        extractor = LearnedPromptedSam3(
            model_id=str(sam_cfg.get("model_id", "facebook/sam3")),
            device=args.device,
            hf_token=hf_token,
            prompt_mode=prompt_mode,
            sparse_num_tokens=int(prompt_cfg.get("num_tokens", 4)),
            sparse_init_std=float(prompt_cfg.get("init_std", 0.02)),
        )
    else:
        extractor = build_frozen_sam_pyramid_extractor(
            model_id=str(sam_cfg.get("model_id", "facebook/sam3")),
            device=args.device,
            hf_token=hf_token,
            include_decoder_semantic_map=bool(sam_cfg.get("include_decoder_semantic_map", False)),
            decoder_prompt_mode=str(sam_cfg.get("decoder_prompt_mode", "full_image_box")),
        )

    # Probe and log feature shapes
    probe_sample = train_samples[0]
    actual_channels = _probe_backbone_channels(extractor, probe_sample, resize_hw)
    probed_feature_map_shapes = _probe_feature_map_shapes(extractor, probe_sample, resize_hw)
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
        "decoder_map_channels": actual_channels.get("decoder_map_channels"),
        "decoder_semantic_map_enabled": bool(sam_cfg.get("include_decoder_semantic_map", False)),
        "decoder_prompt_mode": prompt_mode,
        "decoder_semantic_map_source": str(sam_cfg.get("decoder_semantic_map_source", "")),
        "feature_control": feature_control,
        "prompt_config": prompt_cfg,
        "original_image_size_hw": [int(probe_sample.image.height), int(probe_sample.image.width)],
        "sam_input_size_hw": list(resize_hw) if resize_hw else [int(probe_sample.image.height), int(probe_sample.image.width)],
        "feature_map_shapes_chw": probed_feature_map_shapes,
        "d0_shape_chw": probed_feature_map_shapes.get("decoder_semantic_map"),
        "final_readout_feature_resolution_hw": (
            probed_feature_map_shapes.get("decoder_semantic_map", [None, None, None])[1:]
        ),
        "output_mask_resolution_hw": [
            int(np.asarray(probe_sample.texture_a_mask).shape[0]),
            int(np.asarray(probe_sample.texture_a_mask).shape[1]),
        ],
        "postprocess_eval_resolution_hw": [
            int(np.asarray(probe_sample.texture_a_mask).shape[0]),
            int(np.asarray(probe_sample.texture_a_mask).shape[1]),
        ],
        "expected_paper_channels": {"f2_channels": 256, "f1_channels": 64, "f0_channels": 32},
        "channel_parity_ok": (
            actual_channels.get("f2_channels") == 256
            and actual_channels.get("f1_channels") == 64
            and actual_channels.get("f0_channels") == 32
        ),
    }
    (out_dir / "feature_shapes.json").write_text(json.dumps(feature_shapes, indent=2, sort_keys=True))
    (out_dir / "prompt_config.json").write_text(json.dumps(prompt_cfg, indent=2, sort_keys=True))
    print(f"[info] feature_shapes: {feature_shapes}", flush=True)

    head = _build_head(variant, actual_channels, method_config).to(device)
    head_param_count = _head_param_count(head)
    head_class_name = type(head).__name__
    attention_type = str(getattr(head, "attention_type", ""))
    feature_shapes["head_class_name"] = head_class_name
    feature_shapes["attention_type"] = attention_type
    (out_dir / "feature_shapes.json").write_text(json.dumps(feature_shapes, indent=2, sort_keys=True))
    prompt_param_count = int(extractor.prompt_parameter_count()) if isinstance(extractor, LearnedPromptedSam3) else 0
    feature_sources = _feature_sources_for_variant(variant)
    if is_two_stage:
        head.set_f0_enabled(False)  # stage 1: inner head only

    lr_schedule = str(training_cfg.get("lr_schedule", "constant")).lower()
    lr_min = float(training_cfg.get("lr_min", 0.0))

    optim_params = [p for p in head.parameters() if p.requires_grad]
    if isinstance(extractor, LearnedPromptedSam3):
        optim_params.extend([p for p in extractor.parameters() if p.requires_grad])
    optimizer = torch.optim.AdamW(
        optim_params,
        lr=float(training_cfg.get("lr", 1e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )
    checkpoint_module = torch.nn.ModuleDict({"head": head})
    if isinstance(extractor, LearnedPromptedSam3):
        checkpoint_module["prompt_model"] = extractor

    def _make_scheduler(opt: torch.optim.Optimizer, n_epochs: int) -> "torch.optim.lr_scheduler._LRScheduler | None":
        if lr_schedule == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr_min)
        return None

    scheduler = _make_scheduler(optimizer, stage1_epochs if is_two_stage else total_epochs)
    threshold = float(config.get("evaluation", {}).get("threshold", 0.5))
    prompt_insertion_note = extractor.prompt_state_summary() if isinstance(extractor, LearnedPromptedSam3) else None

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
        wandb_tags = ["monuseg", "test_screen", variant, f"seed{args.seed}"]
        if variant.startswith("d0fpn_"):
            wandb_tags = [
                "monuseg",
                "corrected37",
                "d0_fpn",
                "attention_context",
                "final_epoch_claimable",
                "aug80",
                variant,
                f"seed{args.seed}",
            ]
            if "cap160" in variant:
                wandb_tags.append("capacity160")
            if "task" in attention_type or "tokens" in attention_type:
                wandb_tags.append("task_tokens")
            if "spatial" in attention_type:
                wandb_tags.append("spatial_attention")
            if any(token in attention_type for token in ("residual", "skip", "delta", "correction")):
                wandb_tags.append("residual_skip")
        wandb_run = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            job_type=args.wandb_job_type,
            name=args.run_name,
            tags=wandb_tags,
            config={
                "run_name": args.run_name,
                "variant": variant,
                "variant_label": str(config.get("model", {}).get("variant_label", "")),
                "seed": int(args.seed),
                "eval_split": "test",
                "stage": "TEST_SCREEN",
                "backbone_backend": backbone_backend,
                "paper_headline_safe": feature_shapes["paper_headline_safe"],
                "channel_parity_ok": feature_shapes["channel_parity_ok"],
                "f2_channels": feature_shapes["f2_channels"],
                "f1_channels": feature_shapes["f1_channels"],
                "f0_channels": feature_shapes["f0_channels"],
                "decoder_map_channels": feature_shapes["decoder_map_channels"],
                "decoder_semantic_map_enabled": feature_shapes["decoder_semantic_map_enabled"],
                "decoder_prompt_mode": feature_shapes["decoder_prompt_mode"],
                "head_class_name": head_class_name,
                "attention_type": attention_type,
                "prompt_mode": prompt_mode,
                "prompt_config": prompt_cfg,
                "prompt_insertion_point": prompt_insertion_note,
                "feature_sources": feature_sources,
                "stage1_epochs": stage1_epochs,
                "stage2_epochs": stage2_epochs,
                "eval_every": eval_every,
                "visual_every": visual_every,
                "augmentation_policy": aug_policy,
                "augmentation_description": describe_augmentation_policy(aug_policy),
                "train_selection_policy": split_manifest["train_selection_policy"],
                "train_count": len(train_samples),
                "test_count": len(test_samples),
                "config_path": str(Path(args.config).resolve()),
                "head_param_count": head_param_count,
                "prompt_param_count": prompt_param_count,
                "feature_sources": feature_sources,
                "feature_control": feature_control,
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
            "run_name": args.run_name,
            "variant": variant,
            "variant_label": str(config.get("model", {}).get("variant_label", "")),
            "seed": int(args.seed),
            "eval_split": "test",
            "stage": "TEST_SCREEN",
            "backbone_backend": backbone_backend,
            "head_param_count": head_param_count,
            "head_class_name": head_class_name,
            "attention_type": attention_type,
            "prompt_param_count": prompt_param_count,
            "feature_sources": feature_sources,
            "feature_control": feature_control,
            "train_selection_policy": split_manifest["train_selection_policy"],
            "prompt_mode": prompt_mode,
            "prompt_config": prompt_cfg,
        },
    )

    prompt_stats_csv = out_dir / "prompt_stats.csv"
    prompt_stats_csv.write_text(
        "epoch,prompt_mode,prompt_trainable,prompt_num_tokens,prompt_dim,prompt_param_count,prompt_norm,prompt_delta_norm,prompt_cosine_from_init\n"
    )
    if isinstance(extractor, LearnedPromptedSam3):
        prompt_smoke = {
            "trainability": {
                "sam_trainable_param_count": 0,
                "head_trainable_param_count": head_param_count,
                "prompt_trainable_param_count": prompt_param_count,
                "optimizer_param_count": int(sum(p.numel() for group in optimizer.param_groups for p in group["params"])),
            },
            "constant_prompt": prompt_insertion_note,
            "insertion_point": {
                "category": prompt_insertion_note.get("prompt_category"),
                "tensor_name": prompt_insertion_note.get("prompt_tensor_name"),
                "tensor_shape": prompt_insertion_note.get("prompt_tensor_shape"),
                "trainable_parameter_shape": prompt_insertion_note.get("prompt_trainable_shape"),
                "parameter_count": prompt_insertion_note.get("prompt_param_count"),
                "initialized_from": prompt_insertion_note.get("prompt_initialized_from", prompt_insertion_note.get("prompt_init")),
                "shared_across_images": prompt_insertion_note.get("prompt_shared_across_images"),
                "image_conditioned": prompt_insertion_note.get("prompt_image_conditioned"),
                "gt_derived": prompt_insertion_note.get("prompt_gt_derived"),
            },
            "no_gt_leakage": {
                "uses_gt_mask_prompt": False,
                "uses_gt_box_prompt": False,
                "uses_gt_point_prompt": False,
                "uses_text_from_labels": False,
                "uses_prediction_prompt_at_test": False,
                "uses_test_time_adaptation": False,
            },
        }
        if prompt_mode == "fixed_full_image_box_plus_learned_delta":
            prompt_smoke["equivalence"] = extractor.compare_against_reference(_resize_image(probe_sample.image, resize_hw))
        (out_dir / "prompt_smoke_checks.json").write_text(json.dumps(prompt_smoke, indent=2, sort_keys=True))
        (out_dir / "prompt_insertion_point.json").write_text(json.dumps(prompt_smoke["insertion_point"], indent=2, sort_keys=True))

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    best_test_dice = float("-inf")
    best_test_iou = float("-inf")
    best_test_epoch = -1
    final_epoch_dice = float("nan")
    final_epoch_iou = float("nan")
    final_eval_result = None
    epoch_eval_metrics: dict[int, dict[str, float]] = {}
    final_ckpt = out_dir / "checkpoint.pt"
    ep80_ckpt = out_dir / "checkpoint_ep80.pt"
    ep120_ckpt = out_dir / "checkpoint_ep120.pt"
    best_ckpt = out_dir / "checkpoint_best.pt"
    latest_ckpt = out_dir / "checkpoint_last.pt"
    per_epoch_csv = out_dir / "per_epoch_metrics.csv"
    per_epoch_eval_csv = out_dir / "eval" / "per_epoch_metrics.csv"
    per_epoch_header = "epoch,stage,train_loss,lr,test_dice,test_iou,is_best\n"
    per_epoch_csv.write_text(per_epoch_header)
    per_epoch_eval_csv.write_text(per_epoch_header)

    for epoch in range(1, total_epochs + 1):
        if hasattr(head, "set_epoch"):
            try:
                head.set_epoch(epoch)
            except Exception as exc:
                print(f"[warn] head.set_epoch failed at epoch {epoch}: {exc}", file=sys.stderr)
        # Two-stage transition
        if is_two_stage and epoch == stage1_epochs + 1:
            print(f"[{args.run_name}] Stage 2: enabling F0 residual at epoch {epoch}", flush=True)
            head.set_f0_enabled(True)
            # Re-init optimizer to include new F0 params
            optim_params = [p for p in head.parameters() if p.requires_grad]
            if isinstance(extractor, LearnedPromptedSam3):
                optim_params.extend([p for p in extractor.parameters() if p.requires_grad])
            optimizer = torch.optim.AdamW(
                optim_params,
                lr=float(training_cfg.get("lr", 1e-4)),
                weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
            )
            scheduler = _make_scheduler(optimizer, stage2_epochs)

        if isinstance(extractor, LearnedPromptedSam3):
            train_stats = _train_one_epoch_prompted(
                prompt_model=extractor,
                head=head,
                samples=train_samples,
                resize_hw=resize_hw,
                aug_policy=aug_policy,
                rng=aug_rng,
                optimizer=optimizer,
                loss_fn=bce_dice_loss,
                device=device,
            )
        else:
            train_stream = _streaming_feature_iter(
                extractor, train_samples, resize_hw,
                aug_policy=aug_policy, rng=aug_rng,
                feature_control=feature_control,
                seed=args.seed,
            )
            train_stats = train_one_epoch(
                model=head, dataset=train_stream, optimizer=optimizer,
                loss_fn=bce_dice_loss, device=device,
            )

        should_eval = (epoch == 1) or (epoch % eval_every == 0) or (epoch == total_epochs)
        test_dice = float("nan")
        test_iou = float("nan")
        test_result = None
        if should_eval:
            test_result = run_official_eval(
                extractor=extractor, head=head, samples=test_samples,
                device=device, threshold=threshold, resize_before_sam_hw=resize_hw,
                pyramid_transform=lambda sample_id, pyramid: _apply_decoder_map_control(
                    pyramid,
                    sample_id=str(sample_id),
                    seed=args.seed,
                    control=feature_control,
                ),
            )
            test_dice = float(test_result.dice)
            test_iou = float(test_result.iou)
            epoch_eval_metrics[epoch] = {"dice": test_dice, "iou": test_iou}
            if epoch == total_epochs:
                final_eval_result = test_result

        lr = float(optimizer.param_groups[0]["lr"])
        save_checkpoint(latest_ckpt, model=checkpoint_module, optimizer=optimizer, epoch=epoch)
        if should_eval and test_dice > best_test_dice:
            best_test_dice = test_dice
            best_test_iou = test_iou
            best_test_epoch = epoch
            save_checkpoint(best_ckpt, model=checkpoint_module, optimizer=optimizer, epoch=epoch)
        if should_eval:
            final_epoch_dice = test_dice
            final_epoch_iou = test_iou
        if epoch == total_epochs:
            save_checkpoint(final_ckpt, model=checkpoint_module, optimizer=optimizer, epoch=epoch)
        if epoch == 80 and total_epochs >= 80:
            save_checkpoint(ep80_ckpt, model=checkpoint_module, optimizer=optimizer, epoch=epoch)
        if epoch == 120 and total_epochs >= 120:
            save_checkpoint(ep120_ckpt, model=checkpoint_module, optimizer=optimizer, epoch=epoch)
        if scheduler is not None:
            scheduler.step()

        stage_label = "stage1" if (is_two_stage and epoch <= stage1_epochs) else ("stage2" if is_two_stage else "train")

        # Append to per-epoch CSV (no buffering — readable mid-run)
        per_epoch_row = (
            f"{epoch},{stage_label},{train_stats['loss']:.6f},{lr:.8f},"
            f"{test_dice:.6f},{test_iou:.6f},{int(should_eval and epoch == best_test_epoch)}\n"
        )
        for csv_path in (per_epoch_csv, per_epoch_eval_csv):
            with csv_path.open("a") as _f:
                _f.write(per_epoch_row)

        panel_path = None
        should_visualize = should_eval and ((epoch % visual_every == 0) or (epoch == total_epochs))
        if should_visualize:
            pred = _predict_sample(
                extractor=extractor, head=head, sample=fixed_visual_sample,
                resize_hw=resize_hw, device=device, threshold=threshold,
                feature_control=feature_control, seed=args.seed,
            )
            panel_path = _make_fixed_visual(
                sample=fixed_visual_sample, pred=pred, out_dir=out_dir,
                epoch=epoch, variant=variant, seed=args.seed,
                dice=test_dice, iou=test_iou,
            )
        alpha_val = None
        if hasattr(head, "alpha"):
            try:
                alpha_attr = getattr(head, "alpha")
                if alpha_attr is not None:
                    alpha_val = float(alpha_attr.item() if hasattr(alpha_attr, "item") else float(alpha_attr))
            except Exception:
                alpha_val = None
        prompt_log_dict = {}
        if isinstance(extractor, LearnedPromptedSam3):
            prompt_log_dict = extractor.prompt_state_summary()
            with prompt_stats_csv.open("a") as handle:
                handle.write(
                    f"{epoch},{prompt_log_dict.get('prompt_mode','')},"
                    f"{int(bool(prompt_log_dict.get('prompt_trainable', False)))},"
                    f"{prompt_log_dict.get('prompt_num_tokens', 0)},"
                    f"{prompt_log_dict.get('prompt_dim', 0)},"
                    f"{prompt_log_dict.get('prompt_param_count', 0)},"
                    f"{prompt_log_dict.get('prompt_norm', float('nan'))},"
                    f"{prompt_log_dict.get('prompt_delta_norm', float('nan'))},"
                    f"{prompt_log_dict.get('prompt_cosine_from_init', float('nan'))}\n"
                )
        context_log_dict = {}
        if hasattr(head, "context_state_summary"):
            try:
                context_log_dict = dict(getattr(head, "context_state_summary")() or {})
            except Exception as exc:
                print(f"[warn] attention context summary failed at epoch {epoch}: {exc}", file=sys.stderr)

        if wandb_run is not None:
            try:
                import wandb as wandb_mod
                log_dict = {
                    "epoch": epoch,
                    "train/loss": float(train_stats["loss"]),
                    "train/lr": lr,
                    "provenance/eval_split": "test",
                    "provenance/stage": stage_label,
                }
                if should_eval:
                    log_dict["test/dice"] = test_dice
                    log_dict["test/iou"] = test_iou
                if log_visuals_to_wandb and panel_path is not None:
                    log_dict["test/sample_segmentation"] = wandb_mod.Image(
                        np.asarray(Image.open(panel_path)),
                        caption=(
                            f"{args.run_name} | epoch={epoch} | split=test "
                            f"| sample={fixed_visual_sample.crop_name} "
                            f"| dice={test_dice:.4f} | iou={test_iou:.4f} | variant={variant} | seed={args.seed}"
                        ),
                    )
                if alpha_val is not None:
                    log_dict["model/alpha"] = alpha_val
                for key, value in prompt_log_dict.items():
                    if isinstance(value, (int, float, bool)):
                        log_dict[f"prompt/{key}"] = value
                for key, value in context_log_dict.items():
                    if isinstance(value, (int, float, bool)):
                        log_dict[f"attention/{key}"] = value
                wandb_run.log(log_dict)
            except Exception as exc:
                print(f"[warn] wandb logging failed at epoch {epoch}: {exc}", file=sys.stderr)

        print(
            f"[{args.run_name}] epoch {epoch}/{total_epochs} ({stage_label}) "
            f"loss={train_stats['loss']:.4f}"
            + (f" test_dice={test_dice:.4f} test_iou={test_iou:.4f}" if should_eval else " test=skipped"),
            flush=True,
        )

    # -----------------------------------------------------------------------
    # Save final artifacts
    # -----------------------------------------------------------------------
    semantic_coherence = _write_semantic_coherence_diagnostics(
        extractor=extractor,
        head=head,
        samples=test_samples,
        resize_hw=resize_hw,
        device=device,
        threshold=threshold,
        out_dir=out_dir,
        feature_control=feature_control,
        seed=args.seed,
    )
    final_visuals_path = _write_final_visual_package(
        extractor=extractor,
        head=head,
        samples=test_samples,
        resize_hw=resize_hw,
        device=device,
        threshold=threshold,
        out_dir=out_dir,
        variant=variant,
        seed=args.seed,
        feature_control=feature_control,
    )
    if final_eval_result is not None:
        write_per_sample_csv(out_dir / "eval" / "per_sample_final_epoch.csv", final_eval_result.per_sample_records)
    final_sha = _sha256(final_ckpt)
    best_sha = _sha256(best_ckpt) if best_ckpt.exists() else ""
    (out_dir / "checkpoint_sha256.txt").write_text(final_sha + "\n")
    attention_diagnostics = {
        "head_class_name": head_class_name,
        "attention_type": attention_type,
        "context_summary": {},
        "attention_map_shapes": {},
        "attention_map_keys": [],
    }
    if hasattr(head, "context_state_summary"):
        try:
            attention_diagnostics["context_summary"] = dict(getattr(head, "context_state_summary")() or {})
        except Exception as exc:
            attention_diagnostics["context_summary_error"] = str(exc)
    if hasattr(head, "attention_maps"):
        try:
            attn_maps = getattr(head, "attention_maps")() or {}
            attention_diagnostics["attention_map_keys"] = sorted(attn_maps)
            attention_diagnostics["attention_map_shapes"] = {
                key: list(np.asarray(value).shape) for key, value in attn_maps.items()
            }
        except Exception as exc:
            attention_diagnostics["attention_map_error"] = str(exc)
    (out_dir / "attention_diagnostics.json").write_text(json.dumps(attention_diagnostics, indent=2, sort_keys=True))
    prompt_reload_ok = None
    if isinstance(extractor, LearnedPromptedSam3):
        before = {
            key: value.detach().cpu().clone()
            for key, value in extractor.state_dict().items()
        }
        reloaded = torch.nn.ModuleDict({"head": _build_head(variant, actual_channels, method_config).to(device)})
        reloaded["prompt_model"] = LearnedPromptedSam3(
            model_id=str(sam_cfg.get("model_id", "facebook/sam3")),
            device=args.device,
            hf_token=hf_token,
            prompt_mode=prompt_mode,
            sparse_num_tokens=int(prompt_cfg.get("num_tokens", 4)),
            sparse_init_std=float(prompt_cfg.get("init_std", 0.02)),
        )
        reloaded["prompt_model"].prompt_spec()
        load_checkpoint(final_ckpt, model=reloaded, map_location=str(device), strict=False)
        after = {key: value.detach().cpu().clone() for key, value in reloaded["prompt_model"].state_dict().items()}
        prompt_reload_ok = all(torch.equal(before[key], after[key]) for key in before)

    test_summary = {
        "dataset": "monuseg",
        "variant": variant,
        "variant_label": str(config.get("model", {}).get("variant_label", "")),
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
        "train_selection_policy": split_manifest["train_selection_policy"],
        "eval_every": eval_every,
        "visual_every": visual_every,
        "checkpoint_selection": "final_epoch",
        "fixed_schedule_checkpoint_views": {
            str(epoch): {
                "selected_epoch": epoch,
                "test_dice": epoch_eval_metrics.get(epoch, {}).get("dice", float("nan")),
                "test_iou": epoch_eval_metrics.get(epoch, {}).get("iou", float("nan")),
                "checkpoint_path": str((out_dir / f"checkpoint_ep{epoch}.pt").resolve()) if (out_dir / f"checkpoint_ep{epoch}.pt").exists() else "",
            }
            for epoch in (80, 120, total_epochs)
            if total_epochs >= epoch
        },
        "head_param_count": head_param_count,
        "head_class_name": head_class_name,
        "attention_type": attention_type,
        "prompt_param_count": prompt_param_count,
        "prompt_mode": prompt_mode,
        "prompt_config_path": str((out_dir / "prompt_config.json").resolve()),
        "prompt_stats_path": str(prompt_stats_csv.resolve()),
        "prompt_insertion_point_path": str((out_dir / "prompt_insertion_point.json").resolve()) if prompt_insertion_note is not None else "",
        "prompt_insertion_point": prompt_insertion_note,
        "prompt_reload_ok": prompt_reload_ok,
        "feature_control": feature_control,
        "feature_sources": feature_sources,
        "attention_diagnostics_path": str((out_dir / "attention_diagnostics.json").resolve()),
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
        "resolved_config_path": str((out_dir / "resolved_config.yaml").resolve()),
        "split_manifest_path": str((out_dir / "split_manifest.json").resolve()),
        "train_count": len(train_samples),
        "test_count": len(test_samples),
        "backbone_backend": backbone_backend,
        "head_class_name": head_class_name,
        "attention_type": attention_type,
        "channel_parity_ok": feature_shapes["channel_parity_ok"],
        "paper_headline_safe": feature_shapes["paper_headline_safe"],
        "decoder_semantic_map_enabled": feature_shapes["decoder_semantic_map_enabled"],
        "decoder_prompt_mode": feature_shapes["decoder_prompt_mode"],
        "semantic_coherence": semantic_coherence,
        "feature_shapes_path": str((out_dir / "feature_shapes.json").resolve()),
        "feature_sources": feature_sources,
        "visuals_path": str(final_visuals_path.resolve()),
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
                "test/final_dice": final_epoch_dice,
                "test/final_iou": final_epoch_iou,
                "test/epoch80_dice": epoch_eval_metrics.get(80, {}).get("dice", float("nan")),
                "test/epoch80_iou": epoch_eval_metrics.get(80, {}).get("iou", float("nan")),
                "test/epoch120_dice": epoch_eval_metrics.get(120, {}).get("dice", float("nan")),
                "test/epoch120_iou": epoch_eval_metrics.get(120, {}).get("iou", float("nan")),
                "test/eval_split": "test",
                "checkpoint_sha256": final_sha,
                "channel_parity_ok": feature_shapes["channel_parity_ok"],
                "backbone_backend": backbone_backend,
                "decoder_semantic_map_enabled": feature_shapes["decoder_semantic_map_enabled"],
                "head_param_count": head_param_count,
                "head_class_name": head_class_name,
                "attention_type": attention_type,
                "prompt_param_count": prompt_param_count,
                "prompt_mode": prompt_mode,
                "feature_sources": feature_sources,
            })
            if prompt_insertion_note is not None:
                for key, value in prompt_insertion_note.items():
                    wandb_run.summary[f"prompt/{key}"] = value
            for key, value in attention_diagnostics.get("context_summary", {}).items():
                wandb_run.summary[f"attention/{key}"] = value
            if semantic_coherence:
                for key, value in semantic_coherence.get("mean", {}).items():
                    wandb_run.summary[f"semantic/{key}"] = value
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
