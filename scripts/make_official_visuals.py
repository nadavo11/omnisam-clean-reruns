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
import hashlib
import shlex
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from frozen_sam_readout.data import build_dataset_iter
from frozen_sam_readout.models.compat.source_loader import load_source_checkpoint_head
from frozen_sam_readout.models.registry import build_head
from frozen_sam_readout.sam import build_frozen_sam_pyramid_extractor
from frozen_sam_readout.training.checkpointing import load_checkpoint
from frozen_sam_readout.utils import load_yaml_config
from frozen_sam_readout.visualization import SamplePanels, save_official_visuals

# Registry key → (SamplePanels field name, checkpoint subfolder hint)
_VARIANT_MAP: list[tuple[str, str, str]] = [
    ("a0_fpn2",            "pred_a0",    "A0"),
    ("a2_fpn2_fpn1_refine","pred_a2",    "A2"),
    ("a3_memory",          "pred_a3",    "A3"),
    ("final_staged",       "pred_final", "final_staged"),
]
_DEFAULT_CHANNELS = {"f2_channels": 256, "f1_channels": 64, "f0_channels": 32}


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_official_eval_split(cfg: dict) -> str:
    return str(cfg.get("evaluation", {}).get("split", "test"))


def _build_native_head(variant: str) -> torch.nn.Module:
    kwargs = dict(_DEFAULT_CHANNELS)
    if variant == "a0_fpn2":
        kwargs.pop("f1_channels", None)
        kwargs.pop("f0_channels", None)
    elif variant in {"a2_fpn2_fpn1_refine", "a3_memory"}:
        kwargs.pop("f0_channels", None)
    if variant in {"a3_memory", "a5_all_at_once", "final_staged"}:
        kwargs["memory_tokens"] = 8
    if variant in {"a5_all_at_once", "final_staged"}:
        kwargs["alpha_init"] = 0.05
    if variant == "final_staged":
        kwargs["projection_dim"] = 128
        kwargs["final_layer_zero_init"] = True
    return build_head(variant, **kwargs)


def _load_checkpoint_head(variant: str, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "head_state_dict" in payload:
        return load_source_checkpoint_head(str(checkpoint), device=device)

    head = _build_native_head(variant).to(device)
    load_checkpoint(checkpoint, model=head, map_location=str(device), strict=True)
    return head.eval()


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
    parser.add_argument(
        "--final-staged-only",
        action="store_true",
        help="Load only the canonical final_staged checkpoint and emit final visuals.",
    )
    parser.add_argument("--wandb-project", default=None,
                        help="wandb project name. Omit to skip wandb logging.")
    args = parser.parse_args(argv)

    cfg = load_yaml_config(args.config)
    device = _resolve_device(args.device)
    ckpt_root = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    eval_split = _resolve_official_eval_split(cfg)

    # ── SAM extractor ────────────────────────────────────────────────────────
    sam_cfg = cfg["sam"]
    extractor = build_frozen_sam_pyramid_extractor(
        model_id=str(sam_cfg["model_id"]), device=args.device
    )

    # ── Load variant heads ────────────────────────────────────────────────────
    heads: dict[str, torch.nn.Module] = {}
    checkpoint_paths: dict[str, str] = {}
    if args.final_staged_only:
        variant_map = [item for item in _VARIANT_MAP if item[0] == "final_staged"]
    else:
        variant_map = list(_VARIANT_MAP)

    missing_variants: list[str] = []
    for variant, _field, subfolder in variant_map:
        ckpt = _find_checkpoint(ckpt_root, subfolder)
        if ckpt is not None:
            head = _load_checkpoint_head(variant, ckpt, device)
            checkpoint_paths[variant] = str(ckpt)
            print(f"  Loaded {variant} ← {ckpt}")
        else:
            if args.final_staged_only and variant == "final_staged":
                raise FileNotFoundError(
                    "final_staged checkpoint not found in checkpoint-dir; "
                    "this mode requires the canonical final_staged checkpoint."
                )
            if not args.final_staged_only:
                missing_variants.append(variant)
            checkpoint_paths[variant] = ""
            heads[variant] = None
            continue
        heads[variant] = head

    if missing_variants:
        raise FileNotFoundError(
            "Missing comparison checkpoints for: " + ", ".join(sorted(missing_variants))
        )

    # ── Dataset ───────────────────────────────────────────────────────────────
    ds_cfg = cfg["dataset"]
    resize = ds_cfg.get("resize_before_sam")
    resize_hw = (int(resize["height"]), int(resize["width"])) if resize else None

    ds_kwargs = dict(split=eval_split, limit=args.limit)
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
    provenance = {
        "script": str(Path(__file__).resolve()),
        "command": shlex.join(sys.argv),
        "config_path": str(Path(args.config).resolve()),
        "dataset": str(ds_cfg["name"]),
        "protocol": Path(args.config).stem,
        "generated_image_count": len(all_panels),
        "evaluation_split": eval_split,
        "checkpoint_dir": str(ckpt_root) if ckpt_root is not None else None,
        "checkpoints": {
            variant: {
                "path": path,
                "sha256": _sha256(Path(path)) if path else None,
            }
            for variant, path in checkpoint_paths.items()
        },
        "final_staged_only": bool(args.final_staged_only),
    }
    manifest_path = save_official_visuals(
        sample_panels=all_panels,
        output_dir=args.output_dir,
        dataset=str(ds_cfg["name"]),
        protocol=Path(args.config).stem,
        config=args.config,
        checkpoints=checkpoint_paths,
        provenance=provenance,
        wandb_run=wandb_run,
        wandb_key_prefix="visuals",
    )
    print(f"\nDone. Manifest → {manifest_path}")

    if wandb_run is not None:
        wandb_run.finish()

    return 0


if __name__ == "__main__":
    sys.exit(main())
