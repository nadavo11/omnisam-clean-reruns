# SAM-3 Feature Extractor Equivalence Audit

**Date**: 2026-05-25  
**Status**: PAPER BLOCKER — discrepancy identified; migrated extractor fixed in this audit  
**Auditor**: migration validation pass

---

## 1. Source Extractor (official runs)

### Model loading path

The source repo's official feature extractor for all MoNuSeg and GlaS results is:

```
Sam3CoarseVsFineScaleFeatureExtractor
  └─ Sam3FeatureClusterCoarseToFineGlobalRunner._sam3_runner
       └─ Sam3Runner._ensure_official_backend()
            ├─ from sam3.model.sam3_image_processor import Sam3Processor as OfficialSam3Processor
            └─ Sam3Model.from_pretrained("facebook/sam3")  [official SAM-3 package]
```

**Source file**: `src/rwtd_sam3/models/sam3_coarse_vs_fine_scale_probe.py`  
**Backend source**: `src/rwtd_sam3/models/sam3_runner.py` lines 231–314

### Preprocessing

Image preprocessing uses the **official SAM-3 processor** via:
```python
base_state = official_backend.processor.set_image(image, state={})
```

This calls `Sam3Processor.set_image()` from the official `sam3` package, which internally handles:
- RGB conversion
- Resize to SAM-3 native input resolution (1024×1024 for ViT-H)
- SAM-3 specific pixel normalization (mean/std from the official SAM-3 distribution)
- tensor dtype and device placement

### Feature extraction

```python
backbone_out = base_state["backbone_out"]
feature_levels = backbone_out.get("backbone_fpn")
```

The `backbone_fpn` is extracted as part of `processor.set_image()` which internally calls the full SAM-3 image encoder forward pass. Features are returned as a list of tensors in the order they appear in the FPN (finest to coarsest or vice versa), then sorted by spatial area descending to assign `fpn_2` (coarsest), `fpn_1` (middle), `fpn_0` (finest).

### Key facts

| Property | Value |
|---|---|
| Model ID | `facebook/sam3` |
| Model package | Official `sam3` Python package |
| Processor | `Sam3Processor` from `sam3.model.sam3_image_processor` |
| SAM input resolution | 1024×1024 (SAM-3 default) |
| Normalization | SAM-3 official normalization (ImageNet-like mean/std via official processor) |
| Feature source | `backbone_fpn` (FPN outputs from SAM-3 image encoder) |
| Feature dtype | float32 (cast from model dtype) |
| Feature levels | 3 levels: fpn_2 (coarsest), fpn_1 (middle), fpn_0 (finest) |
| SAM frozen | Yes — no gradients on any SAM parameters |
| Prompt encoder called | No |
| Mask decoder called | No |

---

## 2. Migrated Extractor (initial migration, DEFECTIVE)

The initial migration agent wrote `_Sam3PyramidExtractor` using:

```python
from transformers import AutoModel, AutoProcessor

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id).to(device)
outputs = model.vision_encoder(inputs["pixel_values"])
fpn_outputs = outputs.backbone_fpn or outputs.fpn_features
```

### Discrepancies vs source

| Property | Source | Initial Migration | Impact |
|---|---|---|---|
| Model package | Official `sam3` package | `transformers.AutoModel` | **CRITICAL** |
| Processor | `Sam3Processor` (official) | `AutoProcessor` (HF) | **CRITICAL** |
| Preprocessing | `processor.set_image()` | `AutoProcessor.__call__()` | **CRITICAL** |
| Feature access | `backbone_out["backbone_fpn"]` | `outputs.backbone_fpn` | High |
| Normalization | SAM-3 official | HF processor normalization | **CRITICAL** |

The `transformers` HuggingFace AutoModel path uses HuggingFace-specific preprocessing that may differ in normalization constants, resize interpolation mode, and padding behavior from the official SAM-3 processor. The resulting feature tensors are **not numerically equivalent** and may differ substantially.

---

## 3. Fixed Extractor (corrected migration)

The migrated extractor at `src/frozen_sam_readout/sam/feature_extractor.py` has been updated.

### Corrected `_Sam3PyramidExtractor`

The class now uses the official SAM-3 package as primary path, with `transformers.AutoModel` as a documented fallback:

```python
class _Sam3PyramidExtractor:
    def _ensure_bundle(self):
        # PRIMARY: official sam3 package (matches source repo)
        try:
            from sam3.model.sam3_image_processor import Sam3Processor as OfficialSam3Processor
            from sam3 import Sam3Model
            model = Sam3Model.from_pretrained(self.model_id, **kwargs)
            processor = OfficialSam3Processor(model=model, device=device)
            self._bundle = ("official", torch, model, processor)
            return self._bundle
        except ImportError:
            pass
        # FALLBACK: transformers.AutoModel — different preprocessing, NOT paper-equivalent
        from transformers import AutoModel, AutoProcessor
        ...
        self._bundle = ("transformers", torch, model, processor)
```

The `extract_sam_pyramid` implementation picks the right code path:

- **Official path**: calls `processor.set_image(image, state={})` then reads `backbone_out["backbone_fpn"]`, exactly as in the source repo.
- **Transformers fallback**: calls `model.vision_encoder(inputs["pixel_values"])` and reads `outputs.backbone_fpn`. Marked `paper_headline_safe: false` in the output if used.

---

## 4. Image resize protocol

In the source repo, the **resize to 512×512 (MoNuSeg) or 224×224 (GlaS) is applied to the PIL Image BEFORE passing it to the SAM-3 extractor**:

```python
# From build_glas_feature_cache in monuseg_frozen_feature_mask_head.py:
image_resized = image.resize((resize_w, resize_h), Image.BILINEAR)
_, pyramid = extractor.extract_sam_pyramid(image_resized)
```

The resized image (512×512 or 224×224) is then processed through SAM-3's own internal resize to 1024×1024. So the effective pipeline is:

```
Native image → PIL resize to 512 (or 224) [BILINEAR] → SAM-3 internal resize to 1024 [official SAM processor] → backbone_fpn features
```

This is what "fixed-resize protocol" means and is correctly captured in the official configs.

---

## 5. Feature shapes

For MoNuSeg 512×512 input:

| Level | Shape [C,H,W] | Description |
|---|---|---|
| fpn_2 | [256, 32, 32] (approx) | Coarsest — post-neck embedding |
| fpn_1 | [256, 64, 64] (approx) | Middle FPN level |
| fpn_0 | [256, 128, 128] (approx) | Finest FPN level |

For GlaS 224×224 input:

| Level | Shape [C,H,W] | Description |
|---|---|---|
| fpn_2 | [256, 14, 14] (approx) | Coarsest |
| fpn_1 | [256, 28, 28] (approx) | Middle |
| fpn_0 | [256, 56, 56] (approx) | Finest |

> **Note**: Exact shapes depend on the SAM-3 internal neck/FPN architecture for the specific model checkpoint. The shapes above are estimates based on typical ViT-H SAM feature maps for the stated input resolutions. Confirmed shapes from actual runs are logged in the cluster job outputs (`eval_debug.log`). These must be verified on the cluster before the final paper revision.

---

## 6. Numerical equivalence status

**Full numerical equivalence** (bit-for-bit or floating-point close) between the source and migrated extractor **cannot be verified locally** because:

1. The official SAM-3 package (`sam3`) requires a gated HuggingFace model (`facebook/sam3`) and proprietary weights.
2. The official checkpoints are on the Run:AI cluster at `/storage/nada/outputs/fixed_resize_enhancement/`.
3. No local GPU with SAM-3 weights is available in this environment.

**Shape/statistics verification on cluster** (required before paper freeze):

```bash
# On cluster, run:
python -c "
from frozen_sam_readout.sam.feature_extractor import build_frozen_sam_pyramid_extractor
from PIL import Image
import numpy as np

ext = build_frozen_sam_pyramid_extractor(model_id='facebook/sam3', device='cuda')
img = Image.open('/path/to/monuseg/test_sample.png').convert('RGB')
img_512 = img.resize((512, 512), Image.BILINEAR)
hw, pyr = ext.extract_sam_pyramid(img_512)

for name, feat in sorted(pyr.items()):
    print(f'{name}: shape={feat.shape} mean={feat.mean():.4f} std={feat.std():.4f} min={feat.min():.4f} max={feat.max():.4f}')
"
```

Compare output against the same script run with the source extractor.

---

## 7. Verdict

| Check | Status |
|---|---|
| Model ID matches | ✅ `facebook/sam3` used in both |
| Official SAM-3 package used | ✅ Fixed (primary path now uses official `sam3` package) |
| Preprocessing equivalent | ✅ Fixed (now uses `processor.set_image()`) |
| Feature key names match (`fpn_2/1/0`) | ✅ Both use same names with spatial-area sort |
| Resize before SAM is BILINEAR PIL | ✅ Confirmed in both |
| Augmentation before SAM (training) | ✅ `autosam_dense_v1` ported (see Priority 3) |
| Numerical equivalence verified | ⚠️ **Pending cluster run** |
| SAM frozen | ✅ Both freeze all SAM parameters |
| Prompt/mask decoder unused | ✅ Confirmed in both |

**Blocker status**: The API discrepancy has been fixed. Numerical equivalence on the cluster is the remaining verification step. This should be executed before paper submission.

---

## 8. Action required before paper freeze

```
[ ] Run equivalence check script on cluster (see section 6)
[ ] Compare fpn_2/1/0 mean/std/shape against source extractor output for ≥2 fixed MoNuSeg samples
[ ] If cosine similarity across channels is < 0.99, investigate normalization difference
[ ] Update this document with confirmed shapes and similarity scores
```
