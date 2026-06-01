"""Frozen SAM pyramid feature extractor.

Wraps a frozen SAM image encoder (SAM-3 or SAM-2 / SAM-1 family) and exposes
three named feature levels (``fpn_2``, ``fpn_1``, ``fpn_0``) suitable for the
staged multiscale frozen readout head.

The extractor never calls SAM's prompt encoder or mask decoder. All SAM
parameters have ``requires_grad=False`` after construction; this is enforced by
``assert_sam_frozen``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Protocol, Tuple

import numpy as np
from PIL import Image


LOGGER = logging.getLogger(__name__)


class FrozenSamPyramidExtractorRuntimeError(RuntimeError):
    """Raised when the frozen-backbone pyramid extractor path becomes invalid."""

    def __init__(self, message: str, *, diagnostics: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics or {}


class FrozenSamPyramidExtractor(Protocol):
    """Minimal interface for a frozen SAM pyramid feature extractor."""

    model_id: str

    def extract_sam_pyramid(
        self, image: Image.Image
    ) -> Tuple[Tuple[int, int], Dict[str, np.ndarray]]:
        """Return ``(image_size_hw, {level_name: feature_map[C,H,W]})``."""


def resolve_frozen_sam_backbone_family(model_id: str) -> str:
    """Infer SAM backbone family from a model id."""

    lowered = str(model_id).strip().lower()
    if "sam3" in lowered:
        return "sam3"
    if "sam2" in lowered:
        return "sam2"
    if "sam-vit" in lowered or "sam_vit" in lowered:
        return "sam1"
    raise FrozenSamPyramidExtractorRuntimeError(
        f"Could not infer a supported frozen SAM backbone family from model id '{model_id}'.",
        diagnostics={"model_id": str(model_id)},
    )


def build_frozen_sam_pyramid_extractor(
    *,
    model_id: str,
    device: str = "auto",
    hf_token: Optional[str] = None,
    official_checkpoint_path: Optional[str] = None,
    include_decoder_semantic_map: bool = False,
    decoder_prompt_mode: str = "full_image_box",
) -> FrozenSamPyramidExtractor:
    """Create a frozen SAM pyramid extractor.

    SAM-2 returns ``(image_embed, high_res_feats[0], high_res_feats[1])``
    mapped to ``fpn_2``/``fpn_1``/``fpn_0`` by descending spatial resolution.
    SAM-3 uses the native ``backbone_fpn`` outputs with the same naming.
    SAM-1 only exposes a single ``fpn_2`` level.
    """

    family = resolve_frozen_sam_backbone_family(model_id)
    if family == "sam2":
        return _Sam2PyramidExtractor(
            model_id=model_id,
            device=device,
            hf_token=hf_token,
            official_checkpoint_path=official_checkpoint_path,
        )
    if family == "sam1":
        return _Sam1SingleScaleExtractor(
            model_id=model_id, device=device, hf_token=hf_token
        )
    # SAM-3 path: requires a SAM3-native checkpoint
    return _Sam3PyramidExtractor(
        model_id=model_id,
        device=device,
        hf_token=hf_token,
        official_checkpoint_path=official_checkpoint_path,
        include_decoder_semantic_map=include_decoder_semantic_map,
        decoder_prompt_mode=decoder_prompt_mode,
    )


def _resolve_device(requested_device: str, torch_module: Any) -> str:
    if requested_device == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch_module.cuda.is_available():
        raise FrozenSamPyramidExtractorRuntimeError(
            "CUDA was requested but no CUDA device is available."
        )
    if requested_device not in {"cpu", "cuda"}:
        raise FrozenSamPyramidExtractorRuntimeError(
            f"Unsupported device '{requested_device}'."
        )
    return requested_device


def _freeze_module(module: Any) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    module.eval()


def _find_local_sam3_checkpoint() -> "Optional[Path]":
    """Search common HF cache locations for a local facebook/sam3 checkpoint."""
    import os
    from pathlib import Path

    # HF hub cache layout: <HF_HOME>/hub/models--facebook--sam3/snapshots/<sha>/sam3.pt
    candidates = []
    hf_home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    snap_root = Path(hf_home) / "hub" / "models--facebook--sam3" / "snapshots"
    if snap_root.is_dir():
        for snap in sorted(snap_root.iterdir(), reverse=True):
            p = snap / "sam3.pt"
            if p.exists():
                candidates.append(p)
    # Also check RUNAI cache root
    runai_cache = os.environ.get("RUNAI_CACHE_ROOT")
    if runai_cache:
        alt = Path(runai_cache) / "huggingface" / "hub" / "models--facebook--sam3" / "snapshots"
        if alt.is_dir():
            for snap in sorted(alt.iterdir(), reverse=True):
                p = snap / "sam3.pt"
                if p.exists():
                    candidates.append(p)
    return candidates[0] if candidates else None


class _Sam2PyramidExtractor:
    """Frozen SAM-2 image encoder exposing fpn_2/fpn_1/fpn_0."""

    def __init__(
        self,
        *,
        model_id: str,
        device: str = "auto",
        hf_token: Optional[str] = None,
        official_checkpoint_path: Optional[str] = None,
    ) -> None:
        self.model_id = str(model_id)
        self.requested_device = str(device)
        self.hf_token = hf_token
        self.official_checkpoint_path = official_checkpoint_path
        self._bundle: Optional[Tuple[Any, Any]] = None

    def extract_sam_pyramid(
        self, image: Image.Image
    ) -> Tuple[Tuple[int, int], Dict[str, np.ndarray]]:
        torch_module, predictor = self._ensure_bundle()
        image_array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        with torch_module.inference_mode():
            predictor.set_image(image_array)
        features = getattr(predictor, "_features", None)
        if not isinstance(features, dict) or "image_embed" not in features:
            raise FrozenSamPyramidExtractorRuntimeError(
                "SAM-2 predictor did not expose `_features['image_embed']`."
            )
        image_embed = features["image_embed"]
        high_res = features.get("high_res_feats")
        if not isinstance(high_res, (list, tuple)) or len(high_res) != 2:
            raise FrozenSamPyramidExtractorRuntimeError(
                "SAM-2 predictor must expose exactly two `high_res_feats` levels."
            )
        candidates = [
            ("image_embed", image_embed),
            ("high_res_0", high_res[0]),
            ("high_res_1", high_res[1]),
        ]
        materialized = []
        for source_name, tensor in candidates:
            shape = tuple(int(v) for v in tensor.shape)
            if len(shape) != 4 or shape[0] != 1:
                raise FrozenSamPyramidExtractorRuntimeError(
                    f"SAM-2 level `{source_name}` had unexpected shape {shape}."
                )
            arr = np.asarray(
                tensor[0].detach().cpu().to(torch_module.float32).numpy(),
                dtype=np.float32,
            )
            materialized.append((arr.shape[1] * arr.shape[2], arr, source_name))
        # Sort by spatial area descending: finest first
        materialized.sort(key=lambda x: x[0], reverse=True)
        finest, middle, coarsest = materialized[0][1], materialized[1][1], materialized[2][1]
        pyramid = {"fpn_2": coarsest, "fpn_1": middle, "fpn_0": finest}
        return (int(image_array.shape[0]), int(image_array.shape[1])), pyramid

    def _ensure_bundle(self) -> Tuple[Any, Any]:
        if self._bundle is not None:
            return self._bundle
        try:
            import torch
            from sam2.build_sam import build_sam2_hf
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise FrozenSamPyramidExtractorRuntimeError(
                "SAM-2 path requires Meta's `sam2` package."
            ) from exc
        device = _resolve_device(self.requested_device, torch)
        kwargs: Dict[str, Any] = {"device": device}
        if self.hf_token:
            kwargs["token"] = self.hf_token
        sam_model = build_sam2_hf(self.model_id, **kwargs)
        _freeze_module(sam_model)
        predictor = SAM2ImagePredictor(sam_model)
        self._bundle = (torch, predictor)
        return self._bundle


class _Sam3PyramidExtractor:
    """Frozen SAM-3 image encoder exposing native ``backbone_fpn`` levels.

    Primary path: official ``sam3`` package (matches the source repo used for
    all official results). Fallback: ``transformers.AutoModel`` — functionally
    similar but uses HuggingFace-specific preprocessing that is NOT
    paper-equivalent; outputs from the fallback path are marked
    ``paper_headline_safe: false``.
    """

    def __init__(
        self,
        *,
        model_id: str,
        device: str = "auto",
        hf_token: Optional[str] = None,
        official_checkpoint_path: Optional[str] = None,
        include_decoder_semantic_map: bool = False,
        decoder_prompt_mode: str = "full_image_box",
    ) -> None:
        self.model_id = str(model_id)
        self.requested_device = str(device)
        self.hf_token = hf_token
        self.official_checkpoint_path = official_checkpoint_path
        self.include_decoder_semantic_map = bool(include_decoder_semantic_map)
        self.decoder_prompt_mode = str(decoder_prompt_mode)
        self._bundle: Optional[Tuple] = None  # ("official"|"transformers", torch, ...)
        self._using_official_backend: bool = False

    def extract_sam_pyramid(
        self, image: Image.Image
    ) -> Tuple[Tuple[int, int], Dict[str, np.ndarray]]:
        bundle = self._ensure_bundle()
        rgb = image.convert("RGB")
        if bundle[0] == "official":
            return self._extract_official(rgb, bundle)
        return self._extract_transformers(rgb, bundle)

    def _extract_official(
        self, rgb: Image.Image, bundle: Tuple
    ) -> Tuple[Tuple[int, int], Dict[str, np.ndarray]]:
        _, torch_module, model, processor = bundle
        # Mirrors source repo: Sam3Runner._ensure_official_backend path.
        # processor.set_image populates backbone_fpn via the official SAM-3 forward.
        try:
            model_device = str(next(model.parameters()).device)
        except StopIteration:
            model_device = str(getattr(model, "device", ""))
        autocast_enabled = model_device.startswith("cuda")
        captured_decoder_maps = []

        def _capture_decoder_map(_module: Any, _inputs: Tuple[Any, ...], output: Any) -> None:
            captured_decoder_maps.append(output.detach().cpu().to(torch_module.float32))

        hook_handle = None
        pixel_decoder = getattr(getattr(model, "segmentation_head", None), "pixel_decoder", None)
        if self.include_decoder_semantic_map:
            if pixel_decoder is None:
                raise FrozenSamPyramidExtractorRuntimeError(
                    "Official SAM-3 model does not expose segmentation_head.pixel_decoder."
                )
            hook_handle = pixel_decoder.register_forward_hook(_capture_decoder_map)
        try:
            with torch_module.inference_mode(), torch_module.autocast(
                device_type="cuda",
                dtype=torch_module.bfloat16,
                enabled=autocast_enabled,
            ):
                state = processor.set_image(rgb, state={})
                if self.include_decoder_semantic_map:
                    if self.decoder_prompt_mode == "full_image_box":
                        state = processor.add_geometric_prompt([0.5, 0.5, 1.0, 1.0], True, state)
                    elif self.decoder_prompt_mode == "null_prompt":
                        state = processor.set_text_prompt("visual", state)
                    else:
                        raise FrozenSamPyramidExtractorRuntimeError(
                            f"Unsupported decoder_prompt_mode={self.decoder_prompt_mode!r}."
                        )
        finally:
            if hook_handle is not None:
                hook_handle.remove()
        backbone_out = state.get("backbone_out", {})
        fpn_outputs = backbone_out.get("backbone_fpn")
        if fpn_outputs is None:
            raise FrozenSamPyramidExtractorRuntimeError(
                "Official SAM-3 processor did not populate 'backbone_fpn' in backbone_out."
            )
        image_size, pyramid = self._pack_fpn(fpn_outputs, torch_module, rgb)
        if self.include_decoder_semantic_map:
            if not captured_decoder_maps:
                raise FrozenSamPyramidExtractorRuntimeError(
                    "SAM-3 pixel decoder hook did not capture a decoder semantic map."
                )
            decoder_map = captured_decoder_maps[-1]
            arr = decoder_map[0] if decoder_map.ndim == 4 else decoder_map
            pyramid["decoder_semantic_map"] = np.asarray(arr.numpy(), dtype=np.float32)
        return image_size, pyramid

    def _extract_transformers(
        self, rgb: Image.Image, bundle: Tuple
    ) -> Tuple[Tuple[int, int], Dict[str, np.ndarray]]:
        _, torch_module, model, processor = bundle
        LOGGER.warning(
            "SAM-3 extractor: using transformers.AutoModel fallback — "
            "preprocessing differs from the official SAM-3 package. "
            "Outputs are NOT paper-equivalent. Set paper_headline_safe=false."
        )
        inputs = processor(images=rgb, return_tensors="pt").to(model.device)
        pixel_values = inputs["pixel_values"]
        model_dtype = getattr(model, "dtype", None)
        if model_dtype is not None and getattr(pixel_values, "dtype", None) != model_dtype:
            pixel_values = pixel_values.to(dtype=model_dtype)
        vision_source = self._resolve_transformers_vision_source(model)
        with torch_module.inference_mode():
            outputs = vision_source.get_vision_features(pixel_values=pixel_values)
        fpn_outputs = getattr(outputs, "backbone_fpn", None) or getattr(
            outputs, "fpn_hidden_states", None
        ) or getattr(outputs, "fpn_features", None)
        if fpn_outputs is None:
            raise FrozenSamPyramidExtractorRuntimeError(
                "SAM-3 transformers path: vision features did not return FPN outputs."
            )
        return self._pack_fpn(fpn_outputs, torch_module, rgb)

    @staticmethod
    def _resolve_transformers_vision_source(model: Any) -> Any:
        """Return the object that actually exposes SAM-3 vision features.

        HuggingFace currently returns ``Sam3VideoModel`` for ``facebook/sam3``.
        That wrapper does not expose ``vision_encoder`` at the top level. The
        detector submodule is the one that owns the image encoder and its
        ``get_vision_features`` helper.
        """

        if hasattr(model, "get_vision_features"):
            return model
        detector_model = getattr(model, "detector_model", None)
        if detector_model is not None and hasattr(detector_model, "get_vision_features"):
            return detector_model
        raise FrozenSamPyramidExtractorRuntimeError(
            "SAM-3 transformers path: no vision feature source found on the loaded model."
        )

    @staticmethod
    def _pack_fpn(
        fpn_outputs: Any,
        torch_module: Any,
        rgb: Image.Image,
    ) -> Tuple[Tuple[int, int], Dict[str, np.ndarray]]:
        ordered = sorted(
            ((int(t.shape[-2] * t.shape[-1]), t) for t in fpn_outputs),
            key=lambda x: x[0],
            reverse=True,
        )
        if len(ordered) < 3:
            raise FrozenSamPyramidExtractorRuntimeError(
                f"SAM-3 backbone_fpn returned {len(ordered)} levels; expected ≥3."
            )
        finest, middle, coarsest = ordered[0][1], ordered[1][1], ordered[2][1]

        def _to_np(t: Any) -> np.ndarray:
            arr = t[0] if t.ndim == 4 else t
            return np.asarray(
                arr.detach().cpu().to(torch_module.float32).numpy(), dtype=np.float32
            )

        pyramid = {
            "fpn_2": _to_np(coarsest),
            "fpn_1": _to_np(middle),
            "fpn_0": _to_np(finest),
        }
        return (int(rgb.height), int(rgb.width)), pyramid

    def _ensure_bundle(self) -> Tuple:
        if self._bundle is not None:
            return self._bundle
        try:
            import torch
        except ImportError as exc:
            raise FrozenSamPyramidExtractorRuntimeError("PyTorch is required.") from exc
        device = _resolve_device(self.requested_device, torch)
        kwargs: Dict[str, Any] = {}
        if self.hf_token:
            kwargs["token"] = self.hf_token
        # Primary: official sam3 package — identical to source repo used for official results.
        # Uses build_sam3_image_model (the package API); Sam3Model.from_pretrained does not exist.
        try:
            from sam3.model.sam3_image_processor import Sam3Processor as OfficialSam3Processor
            from sam3 import build_sam3_image_model  # type: ignore[import]

            # Prefer local HF cache over network download.
            local_ckpt = _find_local_sam3_checkpoint()
            if local_ckpt:
                LOGGER.info("SAM-3: using local checkpoint %s", local_ckpt)
                model = build_sam3_image_model(
                    checkpoint_path=str(local_ckpt), load_from_HF=False, device=str(device),
                )
            else:
                LOGGER.info("SAM-3: downloading facebook/sam3 from HuggingFace")
                model = build_sam3_image_model(load_from_HF=True, device=str(device))
            _freeze_module(model)
            processor = OfficialSam3Processor(model, device=str(device), confidence_threshold=0.5)
            self._bundle = ("official", torch, model, processor)
            self._using_official_backend = True
            LOGGER.info("SAM-3 extractor: loaded via official sam3 package on %s.", device)
            return self._bundle
        except ImportError:
            LOGGER.warning(
                "Official `sam3` package not found. Falling back to transformers.AutoModel. "
                "Results will NOT be paper-equivalent (different preprocessing)."
            )
        # Fallback: transformers AutoModel.
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise FrozenSamPyramidExtractorRuntimeError(
                "SAM-3 requires either the official `sam3` package or `transformers`."
            ) from exc
        processor = AutoProcessor.from_pretrained(self.model_id, **kwargs)
        model = AutoModel.from_pretrained(self.model_id, **kwargs).to(device)
        _freeze_module(model)
        self._bundle = ("transformers", torch, model, processor)
        self._using_official_backend = False
        return self._bundle


class _Sam1SingleScaleExtractor:
    """Original SAM (ViT-H/L/B) exposing image embeddings as ``fpn_2`` only."""

    def __init__(
        self,
        *,
        model_id: str,
        device: str = "auto",
        hf_token: Optional[str] = None,
    ) -> None:
        self.model_id = str(model_id)
        self.requested_device = str(device)
        self.hf_token = hf_token
        self._bundle: Optional[Tuple[Any, Any, Any]] = None

    def extract_sam_pyramid(
        self, image: Image.Image
    ) -> Tuple[Tuple[int, int], Dict[str, np.ndarray]]:
        torch_module, processor, model = self._ensure_bundle()
        rgb = image.convert("RGB")
        inputs = processor(images=rgb, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(model.device)
        with torch_module.inference_mode():
            embeddings = model.get_image_embeddings(pixel_values)
        feature = np.asarray(
            embeddings[0].detach().cpu().to(torch_module.float32).numpy(),
            dtype=np.float32,
        )
        return (int(rgb.height), int(rgb.width)), {"fpn_2": feature}

    def _ensure_bundle(self) -> Tuple[Any, Any, Any]:
        if self._bundle is not None:
            return self._bundle
        try:
            import torch
            from transformers import SamModel, SamProcessor
        except ImportError as exc:
            raise FrozenSamPyramidExtractorRuntimeError(
                "SAM-1 path requires `transformers` with `SamModel`."
            ) from exc
        device = _resolve_device(self.requested_device, torch)
        kwargs: Dict[str, Any] = {}
        if self.hf_token:
            kwargs["token"] = self.hf_token
        processor = SamProcessor.from_pretrained(self.model_id, **kwargs)
        model = SamModel.from_pretrained(self.model_id, **kwargs).to(device)
        _freeze_module(model)
        self._bundle = (torch, processor, model)
        return self._bundle
