"""Official comparison-grid visualiser.

Outputs three artefacts for every evaluated sample:

1. **LaTeX panel hierarchy** — one folder per sample with individually-named
   panel images. Numeric prefixes make ``\\foreach`` trivial in LaTeX:

       panels/{sample_id}/
           00_image.png
           01_gt.png
           02_pred_a0.png
           03_pred_a2.png
           04_pred_a3.png
           05_pred_final.png
           06_error.png

2. **QC grids** — clean horizontal strips (no text, dark gap between panels)
   for fast human qualitative validation:

       grids/grid_{n:02d}_{sample_id}.png

3. **wandb logging** — each grid streamed as ``wandb.Image`` under the key
   ``visuals/{dataset}/grid_{n:02d}``. Optional; skipped if wandb is not
   installed or not initialised.

A ``visuals_manifest.json`` records every sample's panel paths, checkpoint
sources, error-color mapping, and generation timestamp.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .error_maps import ERROR_COLOR_MAP, binary_error_map
from .grids import compose_strip
from .overlay import render_single_mask_overlay, render_gt_overlay

LOGGER = logging.getLogger(__name__)

# Ordered column spec — (panel_key, filename_prefix, display_label)
# label is kept internally for manifest only; never written on the image.
COLUMN_SPEC: list[tuple[str, str, str]] = [
    ("image",       "00_image",       "Input"),
    ("gt",          "01_gt",          "GT"),
    ("pred_a0",     "02_pred_a0",     "A0"),
    ("pred_a2",     "03_pred_a2",     "A2"),
    ("pred_a3",     "04_pred_a3",     "A3"),
    ("pred_final",  "05_pred_final",  "Final"),
    ("error",       "06_error",       "Error"),
]


@dataclass
class SamplePanels:
    """All panels for one evaluated sample."""
    sample_id: str
    image:       np.ndarray        # RGB uint8 [H,W,3]
    gt:          np.ndarray        # binary bool/uint8 [H,W]
    pred_a0:     Optional[np.ndarray] = None
    pred_a2:     Optional[np.ndarray] = None
    pred_a3:     Optional[np.ndarray] = None
    pred_final:  Optional[np.ndarray] = None

    def build_error(self) -> Optional[np.ndarray]:
        if self.pred_final is not None:
            return binary_error_map(self.pred_final, self.gt)
        return None

    def ordered_panels(self) -> list[tuple[str, str, np.ndarray]]:
        """Return (key, filename_prefix, array) for every present column."""
        pil_image = Image.fromarray(self.image)
        error = self.build_error()

        # Prediction colour cycle — matches old-repo palette order
        _PRED_COLORS = [
            (230,  56,  70),   # A0 — red
            ( 55, 120, 235),   # A2 — blue
            ( 46, 125,  50),   # A3 — green
            (244, 162,  97),   # Final — orange
        ]

        def _overlay(mask: np.ndarray, color: tuple) -> np.ndarray:
            return np.asarray(render_single_mask_overlay(pil_image, mask, color), dtype=np.uint8)

        candidates = {
            "image":      self.image,
            "gt":         np.asarray(render_gt_overlay(pil_image, self.gt), dtype=np.uint8),
            "pred_a0":    _overlay(self.pred_a0,    _PRED_COLORS[0]) if self.pred_a0    is not None else None,
            "pred_a2":    _overlay(self.pred_a2,    _PRED_COLORS[1]) if self.pred_a2    is not None else None,
            "pred_a3":    _overlay(self.pred_a3,    _PRED_COLORS[2]) if self.pred_a3    is not None else None,
            "pred_final": _overlay(self.pred_final, _PRED_COLORS[3]) if self.pred_final is not None else None,
            "error":      error,
        }
        out = []
        for key, prefix, _label in COLUMN_SPEC:
            arr = candidates[key]
            if arr is not None:
                out.append((key, prefix, arr))
        return out


@dataclass
class VisualsManifest:
    dataset: str
    protocol: str
    config: str
    checkpoints: Dict[str, str]
    samples: List[Dict[str, Any]] = field(default_factory=list)
    error_color_map: Dict[str, list] = field(default_factory=lambda: {
        k: list(v) for k, v in ERROR_COLOR_MAP.items()
    })
    renderer_version: str = "official_visuals.py::save_official_visuals"
    generated_at: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        if not d["generated_at"]:
            d["generated_at"] = datetime.now(timezone.utc).isoformat()
        return d


def save_official_visuals(
    *,
    sample_panels: Sequence[SamplePanels],
    output_dir: str | Path,
    dataset: str,
    protocol: str,
    config: str,
    checkpoints: Dict[str, str],
    wandb_run: Any = None,
    wandb_key_prefix: str = "visuals",
) -> Path:
    """Save panels, QC grids, manifest; optionally log to wandb.

    Args:
        sample_panels: Ordered sequence of ``SamplePanels`` (one per sample).
        output_dir: Root output directory. Created if absent.
        dataset: Dataset name for manifest (e.g. ``"monuseg"``).
        protocol: Protocol name for manifest (e.g. ``"monuseg_strict_512"``).
        config: Path to the dataset config YAML used.
        checkpoints: ``{variant_key: checkpoint_path}`` dict for manifest.
        wandb_run: Active ``wandb.Run`` object, or ``None`` to skip wandb logging.
        wandb_key_prefix: Prefix for wandb image keys.

    Returns:
        Path to the written ``visuals_manifest.json``.
    """
    out = Path(output_dir)
    panels_dir = out / "panels"
    grids_dir = out / "grids"
    panels_dir.mkdir(parents=True, exist_ok=True)
    grids_dir.mkdir(parents=True, exist_ok=True)

    manifest = VisualsManifest(
        dataset=dataset,
        protocol=protocol,
        config=config,
        checkpoints=checkpoints,
    )

    for n, sp in enumerate(sample_panels):
        # ── 1. LaTeX panel hierarchy ──────────────────────────────────────────
        sample_panel_dir = panels_dir / sp.sample_id
        sample_panel_dir.mkdir(parents=True, exist_ok=True)

        ordered = sp.ordered_panels()
        panel_paths: dict[str, str] = {}
        for key, prefix, arr in ordered:
            fname = f"{prefix}.png"
            fpath = sample_panel_dir / fname
            Image.fromarray(np.asarray(arr, dtype=np.uint8)).save(fpath)
            panel_paths[key] = str(fpath.relative_to(out))

        # ── 2. QC grid ────────────────────────────────────────────────────────
        strip = compose_strip([arr for _, _, arr in ordered])
        grid_name = f"grid_{n:02d}_{sp.sample_id}.png"
        grid_path = grids_dir / grid_name
        strip.save(grid_path)

        # ── 3. wandb ─────────────────────────────────────────────────────────
        if wandb_run is not None:
            try:
                import wandb as _wandb
                key = f"{wandb_key_prefix}/{dataset}/grid_{n:02d}"
                wandb_run.log({key: _wandb.Image(strip, caption=sp.sample_id)})
                LOGGER.debug("Logged %s to wandb.", key)
            except Exception as exc:
                LOGGER.warning("wandb logging failed for sample %s: %s", sp.sample_id, exc)

        manifest.samples.append({
            "n": n,
            "sample_id": sp.sample_id,
            "panels": panel_paths,
            "grid": str(grid_path.relative_to(out)),
        })
        LOGGER.info("Saved sample %d/%d (%s)", n + 1, len(sample_panels), sp.sample_id)

    # ── 4. Manifest ───────────────────────────────────────────────────────────
    manifest_dict = manifest.to_dict()
    manifest_path = out / "visuals_manifest.json"
    manifest_path.write_text(json.dumps(manifest_dict, indent=2))
    LOGGER.info("Wrote manifest → %s", manifest_path)

    # ── 5. LaTeX snippet ─────────────────────────────────────────────────────
    _write_latex_snippet(out, sample_panels, manifest)

    return manifest_path


def _write_latex_snippet(
    out: Path,
    sample_panels: Sequence[SamplePanels],
    manifest: VisualsManifest,
) -> None:
    """Write a ready-to-\\input LaTeX figure skeleton."""
    cols = [spec for spec in COLUMN_SPEC
            if any(sp.ordered_panels() and spec[0] in {k for k, _, _ in sp.ordered_panels()}
                   for sp in sample_panels)]
    col_labels = " & ".join(f"\\textbf{{{spec[2]}}}" for spec in cols)
    n_cols = len(cols)

    lines: list[str] = [
        "% Auto-generated by official_visuals.py — safe to \\input{} from your main .tex",
        f"% Dataset: {manifest.dataset}  Protocol: {manifest.protocol}",
        f"% Generated: {manifest.generated_at}",
        "",
        "\\begin{figure}[htbp]",
        "  \\centering",
        f"  \\setlength{{\\tabcolsep}}{{1pt}}",
        f"  \\renewcommand{{\\arraystretch}}{{0.5}}",
        f"  \\begin{{tabular}}{{{'c' * n_cols}}}",
        f"    {col_labels} \\\\[2pt]",
    ]

    for entry in manifest.samples:
        sid = entry["sample_id"]
        panel_paths = entry["panels"]
        cells = []
        for spec_key, _prefix, _label in cols:
            rel = panel_paths.get(spec_key, "")
            if rel:
                cells.append(f"\\includegraphics[width=0.13\\linewidth]{{panels/{sid}/{Path(rel).name}}}")
            else:
                cells.append("\\phantom{x}")
        lines.append("    " + " & ".join(cells) + " \\\\[1pt]")

    lines += [
        "  \\end{tabular}",
        f"  \\caption{{\\textbf{{{manifest.dataset.upper()} comparison.}} "
        f"Columns: {', '.join(s[2] for s in cols)}.}}",
        "  \\label{fig:" + manifest.dataset + "_comparison}",
        "\\end{figure}",
    ]

    snippet_path = out / "figure_grid.tex"
    snippet_path.write_text("\n".join(lines) + "\n")
    LOGGER.info("Wrote LaTeX snippet → %s", snippet_path)
