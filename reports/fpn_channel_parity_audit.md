# FPN Channel Parity Audit

**Date:** 2026-05-26  
**Context:** First clean E1 matrix (monuseg_E1_A{0,2,3}_s{0,1,2}_clean)

---

## Diagnosis

### Why HF SAM-3 fallback returns 256 channels for all FPN levels

The official SAM-3 model uses a hierarchical Vision Transformer backbone with a Feature Pyramid
Network (FPN) that outputs **decreasing channel counts** at finer spatial levels:

| FPN Level | Spatial (approx) | Official SAM-3 channels | HF Transformers fallback channels |
|---|---|---:|---:|
| fpn_2 (coarsest) | ~32×32 | **256** | 256 |
| fpn_1 (medium) | ~64×64 | **64** | 256 |
| fpn_0 (finest) | ~128×128 | **32** | 256 |

The HuggingFace `transformers.AutoModel` loads `facebook/sam3` as `Sam3VideoModel`.
The `detector_model.get_vision_features()` call returns FPN outputs where all levels
use the full transformer hidden dimension (256), not the laterally-projected smaller dims
from the official SAM-3 package. This is a fundamental architecture difference in how
the two backends expose intermediate features.

### Probe evidence from first clean matrix

```
[info] backbone channel probe: {'f2_channels': 256, 'f1_channels': 256, 'f0_channels': 256}
```
(Seen in logs for all A2/A3 jobs after the channel probe fix in ec2f540)

### Impact on A2 and A3

- **A0** only uses `fpn_2` (256ch) — **unaffected by this discrepancy**.
- **A2** uses `fpn_2` + `fpn_1`: ran with `f1_channels=256` (paper: 64). The FPN refine
  head is larger than intended. Results are consistent across seeds but architecturally different.
- **A3** uses `fpn_2` + `fpn_1`: same as A2. Memory attention block + refine head ran with
  non-paper channel layout.

---

## Options

### Option A — Use Official SAM-3 Package (Preferred for Paper)

- Install the `sam3` package from `https://github.com/facebookresearch/sam3` in the cluster image
  or pre-download the checkpoint to cluster PVC.
- The feature extractor already has the official path implemented and will auto-detect it.
- Expected behavior after install: channel probe returns `{f2: 256, f1: 64, f0: 32}` and
  `feature_shapes.json` will show `channel_parity_ok: true, paper_headline_safe: true`.

**Status:** Not yet available on the cluster (official `sam3` package import fails).
When available, re-run with the same runner (no code changes needed; probe auto-adapts).

### Option B — HF Fallback with Explicit HF256 Variant Names (Acceptable for Now)

Keep the HF fallback but:
1. Rename the output run variants in reports to `A2_HF256` / `A3_HF256` to make the
   channel layout explicit.
2. Do NOT call them paper-equivalent A2/A3.
3. Document the variant rename in the decision report.
4. The `feature_shapes.json` already records `channel_parity_ok: false` and
   `paper_headline_safe: false` — this is the machine-readable flag.

**Status:** Implemented. All v2 runs will write `feature_shapes.json` with these fields.

---

## Current Runner Behavior (Post-Fix)

After commit ec2f540, the training script:
1. Probes actual channel counts from first sample via `_probe_backbone_channels()`
2. Builds the head with the actual channel sizes (no crash)
3. Writes `feature_shapes.json` to `outputs/<run_name>/feature_shapes.json` with:
   - `backbone_backend`: "official" | "transformers_hf_fallback"
   - `paper_headline_safe`: bool
   - `channel_parity_ok`: bool
   - `f2_channels`, `f1_channels`, `f0_channels`: actual values
   - `expected_paper_channels`: {"f2_channels": 256, "f1_channels": 64, "f0_channels": 32}
4. Logs `channel_parity_ok` and `backbone_backend` to W&B run config

---

## Decision for v2 E1 Reruns

- Proceed with HF fallback (Option B) since official SAM-3 is unavailable on cluster.
- v2 runs will be labeled internally as using `transformers_hf_fallback` with `channel_parity_ok=false`.
- Results will be reported as "HF-fallback channel layout (f1=f0=256)" in the decision report.
- The val-metric trend (A0 < A2 < A3) is expected to hold and is architecturally meaningful
  as an ablation even if channel sizes differ from the paper.
- When official SAM-3 becomes available, a v3 clean matrix can be run to get paper-equivalent numbers.

---

## Regression Test

`tests/test_e1_e2_split_contract.py::TestFeatureShapesArtifact` verifies:
- `feature_shapes.json` contains required keys
- `channel_parity_ok=False` when HF fallback is used with uniform 256ch
