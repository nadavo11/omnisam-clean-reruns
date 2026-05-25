#!/usr/bin/env python
"""Generate official comparison visuals.

Outputs per sample:
  panels/{sample_id}/{00_image, 01_gt, 02_pred_a0, ...}.png  ← LaTeX-ready
  grids/grid_{n:02d}_{sample_id}.png                         ← QC strip
  visuals_manifest.json                                       ← full provenance
  figure_grid.tex                                             ← ready-to-\\input LaTeX

Optionally streams grids to wandb with --wandb-project.

Usage example:
  python scripts/make_official_visuals.py \\
    --config configs/official/monuseg_strict_512.yaml \\
    --checkpoint-dir /storage/nada/outputs/fixed_resize_enhancement/migrated_heads \\
    --output-dir outputs/official_visuals/monuseg \\
    --limit 14 \\
    --device cuda \\
    --wandb-project omniSAM-visuals
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from frozen_sam_readout.data import build_dataset_iter
from frozen_sam_readout.models.compat.source_loader import load_source_checkpoint_head
from frozen_sam_readout.sam import build_frozen_sam_pyramid_extractor
from frozen_sam_readout.utils import load_yaml_config
from frozen_sam_readout.visualization import SamplePanels, save_official_visuals

# Registry key → (SamplePanels field name, checkpoint subfolder hint)
_VARIANT_MAP: list[tuple[str, str, str]] = [
    ("a0_fpn2",            "pred_a0",    "A0"),
    ("a2_fpn2_fpn1_refine","pred_a2",    "A2"),
    ("a3_memory",          "pred_a3",    "A3"),
    ("final_staged",       "pred_final", "final_staged"),
]


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _find_checkpoint(ckpt_dir: Path | None, subfolder: str) -> Path | None:
    if ckpt_dir is None:
        return None
    for candidate in [
        ckpt_dir / subfolder / "checkpoint.pt",
        ckpt_dir / subfolder / "checkpoint_final.pt",
        ckpt_dir / subfolder / "stage2.pt",
        ckpt_dir / subfolder / "stage1.pt",
        ckpt_dir / f"{subfolder}.pt",
    ]:
        if candidate.exists():
            return candidate
    return None


def _predict(
    head: torch.nn.Module,
    pyramid: dict[str, np.ndarray],
    gt_shape: tuple[int, int],
    device: torch.device,
    threshold: float,
) -> np.ndarray:
    head.eval()
    with torch.inference_mode():
        tensors = {
            k: torch.from_numpy(np.ascontiguousarray(v)).float().unsqueeze(0).to(device)
            for k, v in pyramid.items()
        }
        logits = head(tensors)
        logits_up = F.interpolate(logits, size=gt_shape, mode="bilinear", align_corners=False)
        return (torch.sigmoid(logits_up)[0, 0].cpu().numpy() > threshold).astype(bool)


def _init_wandb(project: str, config: dict) -> Any:
    try:
        import wandb
        run = wandb.init(project=project, config=config, job_type="visuals")
        return run
    except Exception as exc:
        print(f"[warn] wandb init failed: {exc}", file=sys.stderr)
        return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate official comparison visuals.")
    parser.add_argument("--config", required=True, help="Dataset/protocol YAML config.")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Root directory containing per-variant checkpoint subfolders.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--limit", type=int, default=14, help="Max samples to visualise.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--wandb-project", default=None,
                        help="wandb project name. Omit to skip wandb logging.")
    args = parser.parse_args(argv)

    cfg = load_yaml_config(args.config)
    device = _resolve_device(args.device)
    ckpt_root = Path(args.checkpoint_dir) if args.checkpoint_dir else None

    # ── SAM extractor ────────────────────────────────────────────────────────
    sam_cfg = cfg["sam"]
    extractor = build_frozen_sam_pyramid_extractor(
        model_id=str(sam_cfg["model_id"]), device=args.device
    )

    # ── Load variant heads ────────────────────────────────────────────────────
    heads: dict[str, torch.nn.Module] = {}
    checkpoint_paths: dict[str, str] = {}
    for variant, _field, subfolder in _VARIANT_MAP:
        ckpt = _find_checkpoint(ckpt_root, subfolder)
        if ckpt is not None:
            head = load_source_checkpoint_head(str(ckpt), device=device)
            checkpoint_paths[variant] = str(ckpt)
            print(f"  Loaded {variant} ← {ckpt}")
        else:
            print(f"  [skip] no checkpoint found for {variant} — skipping variant")
            checkpoint_paths[variant] = ""
            heads[variant] = None
            continue
        heads[variant] = head

    # ── Dataset ───────────────────────────────────────────────────────────────
    ds_cfg = cfg["dataset"]
    resize = ds_cfg.get("resize_before_sam")
    resize_hw = (int(resize["height"]), int(resize["width"])) if resize else None

    ds_kwargs = dict(split=ds_cfg.get("split", "test"), limit=args.limit)
    root_or_name = ds_cfg.get("root_or_name")
    if root_or_name:
        ds_kwargs["dataset_root"] = root_or_name
    samples = list(build_dataset_iter(str(ds_cfg["name"]), **ds_kwargs))

    # ── wandb ─────────────────────────────────────────────────────────────────
    wandb_run = None
    if args.wandb_project:
        wandb_run = _init_wandb(
            project=args.wandb_project,
            config={"config": args.config, "limit": args.limit, "device": args.device},
        )

    # ── Build SamplePanels per sample ─────────────────────────────────────────
    all_panels: list[SamplePanels] = []
    for si, sample in enumerate(samples):
        sample_id = str(getattr(sample, "sample_id",
                        getattr(sample, "crop_name", f"sample_{si:03d}")))
        image = sample.image.convert("RGB")
        gt = np.asarray(sample.texture_a_mask, dtype=bool)

        if resize_hw is not None:
            inp = image.resize((resize_hw[1], resize_hw[0]), Image.BILINEAR)
        else:
            inp = image

        _, pyramid = extractor.extract_sam_pyramid(inp)

        preds: dict[str, np.ndarray] = {}
        for variant, field_name, _sub in _VARIANT_MAP:
            h = heads.get(variant)
            if h is None:
                continue
            preds[field_name] = _predict(h, pyramid, gt.shape, device, args.threshold)

        all_panels.append(SamplePanels(
            sample_id=sample_id,
            image=np.asarray(image, dtype=np.uint8),
            gt=gt,
            pred_a0=preds.get("pred_a0"),
            pred_a2=preds.get("pred_a2"),
            pred_a3=preds.get("pred_a3"),
            pred_final=preds.get("pred_final"),
        ))
        print(f"  [{si+1}/{len(samples)}] {sample_id}")

    # ── Save everything ───────────────────────────────────────────────────────
    manifest_path = save_official_visuals(
        sample_panels=all_panels,
        output_dir=args.output_dir,
        dataset=str(ds_cfg["name"]),
        protocol=Path(args.config).stem,
        config=args.config,
        checkpoints=checkpoint_paths,
        wandb_run=wandb_run,
        wandb_key_prefix="visuals",
    )
    print(f"\nDone. Manifest → {manifest_path}")

    if wandb_run is not None:
        wandb_run.finish()

    return 0


if __name__ == "__main__":
    sys.exit(main())
