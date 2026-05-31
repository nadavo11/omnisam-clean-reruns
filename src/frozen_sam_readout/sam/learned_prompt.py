from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

from .feature_extractor import (
    _Sam3PyramidExtractor,
    _find_local_sam3_checkpoint,
    _freeze_module,
    _resolve_device,
)


@dataclass
class Sam3PromptSpec:
    mode: str
    shared_across_images: bool = True
    image_conditioned: bool = False
    gt_derived: bool = False
    trainable: bool = False
    prompt_init: str = "fixed"
    prompt_num_tokens: int = 0
    prompt_dim: int = 0


class LearnedPromptedSam3(torch.nn.Module):
    """Frozen SAM3 forward path with trainable constant prompt controls."""

    def __init__(
        self,
        *,
        model_id: str = "facebook/sam3",
        device: str = "auto",
        hf_token: Optional[str] = None,
        prompt_mode: str = "fixed_full_image_box",
        sparse_num_tokens: int = 4,
        sparse_init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.model_id = str(model_id)
        self.requested_device = str(device)
        self.hf_token = hf_token
        self.prompt_mode = str(prompt_mode)
        self.sparse_num_tokens = int(sparse_num_tokens)
        self.sparse_init_std = float(sparse_init_std)
        self._bundle: Optional[tuple[Any, Any, Any]] = None
        self.transform = None
        self.register_parameter("learned_sparse_tokens", None)
        self.register_parameter("learned_delta", None)
        self._spec: Optional[Sam3PromptSpec] = None
        self._initial_prompt_snapshot: Optional[torch.Tensor] = None

    @property
    def device(self) -> torch.device:
        _, model, _ = self._ensure_bundle()
        return next(model.parameters()).device

    def prompt_spec(self) -> Sam3PromptSpec:
        if self._spec is None:
            self._ensure_prompt_parameters()
        assert self._spec is not None
        return self._spec

    def prompt_parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def prompt_state_summary(self) -> dict[str, Any]:
        spec = self.prompt_spec()
        payload = {
            "prompt_mode": spec.mode,
            "prompt_trainable": bool(spec.trainable),
            "prompt_num_tokens": int(spec.prompt_num_tokens),
            "prompt_dim": int(spec.prompt_dim),
            "prompt_param_count": int(self.prompt_parameter_count()),
            "prompt_init": spec.prompt_init,
            "prompt_shared_across_images": bool(spec.shared_across_images),
            "prompt_image_conditioned": bool(spec.image_conditioned),
            "prompt_gt_derived": bool(spec.gt_derived),
        }
        if self.learned_sparse_tokens is not None:
            prompt = self.learned_sparse_tokens.detach()
            payload["prompt_norm"] = float(prompt.norm().item())
            if self._initial_prompt_snapshot is not None:
                init = self._initial_prompt_snapshot.to(prompt.device)
                payload["prompt_cosine_from_init"] = self._safe_cosine(prompt, init)
        if self.learned_delta is not None:
            delta = self.learned_delta.detach()
            payload["prompt_delta_norm"] = float(delta.norm().item())
            if self._initial_prompt_snapshot is not None:
                init = self._initial_prompt_snapshot.to(delta.device)
                payload["prompt_cosine_from_init"] = self._safe_cosine(delta, init)
        return payload

    @staticmethod
    def _safe_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        flat_a = a.reshape(-1)
        flat_b = b.reshape(-1)
        norm_a = float(flat_a.norm().item())
        norm_b = float(flat_b.norm().item())
        if norm_a == 0.0 and norm_b == 0.0:
            return 1.0
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(torch.nn.functional.cosine_similarity(flat_a[None], flat_b[None], dim=1).item())

    def forward_pyramid(
        self,
        image: Image.Image,
        *,
        capture_numpy: bool = False,
    ) -> dict[str, torch.Tensor] | dict[str, np.ndarray]:
        self._ensure_prompt_parameters()
        torch_mod, model, _ = self._ensure_bundle()
        rgb = image.convert("RGB")
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=str(self.device).startswith("cuda"),
        ):
            state = self._set_image(rgb)
            state = self._apply_prompt(state)
            backbone_out = state["backbone_out"]["backbone_fpn"]
            ordered = sorted(
                ((int(t.shape[-2] * t.shape[-1]), t) for t in backbone_out),
                key=lambda x: x[0],
                reverse=True,
            )
            finest, middle, coarsest = ordered[0][1], ordered[1][1], ordered[2][1]
            decoder_map = self._run_forward_grounding_with_hook(state)
        pyramid_t = {
            "fpn_2": coarsest[0].detach().to(torch.float32),
            "fpn_1": middle[0].detach().to(torch.float32),
            "fpn_0": finest[0].detach().to(torch.float32),
            "decoder_semantic_map": decoder_map[0].to(torch.float32),
        }
        if not capture_numpy:
            return pyramid_t
        return {
            key: np.asarray(value.detach().cpu().numpy(), dtype=np.float32)
            for key, value in pyramid_t.items()
        }

    def extract_sam_pyramid(
        self,
        image: Image.Image,
    ) -> tuple[tuple[int, int], dict[str, np.ndarray]]:
        rgb = image.convert("RGB")
        return (int(rgb.height), int(rgb.width)), self.forward_pyramid(rgb, capture_numpy=True)

    def compare_against_reference(self, image: Image.Image) -> dict[str, float]:
        if self.prompt_mode != "fixed_full_image_box_plus_learned_delta":
            raise ValueError("Equivalence check is only valid for fixed_full_image_box_plus_learned_delta")
        extractor = _Sam3PyramidExtractor(
            model_id=self.model_id,
            device="cuda" if self.device.type == "cuda" else "cpu",
            hf_token=self.hf_token,
            include_decoder_semantic_map=True,
            decoder_prompt_mode="full_image_box",
        )
        _, ref_np = extractor.extract_sam_pyramid(image)
        ours = self.forward_pyramid(image, capture_numpy=False)["decoder_semantic_map"].detach().cpu()
        ref = torch.from_numpy(ref_np["decoder_semantic_map"])
        a = ours.reshape(-1)
        b = ref.reshape(-1)
        return {
            "max_abs_diff": float((a - b).abs().max().item()),
            "mean_abs_diff": float((a - b).abs().mean().item()),
            "cosine_similarity": float(torch.nn.functional.cosine_similarity(a[None], b[None], dim=1).item()),
        }

    def _ensure_bundle(self) -> tuple[Any, Any, Any]:
        if self._bundle is not None:
            return self._bundle
        from sam3 import build_sam3_image_model

        device = _resolve_device(self.requested_device, torch)
        local_ckpt = _find_local_sam3_checkpoint()
        if local_ckpt:
            model = build_sam3_image_model(
                checkpoint_path=str(local_ckpt),
                load_from_HF=False,
                device=str(device),
            )
        else:
            model = build_sam3_image_model(load_from_HF=True, device=str(device))
        _freeze_module(model)
        self.transform = v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(1008, 1008)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self._bundle = (torch, model, model._get_dummy_prompt)
        return self._bundle

    def _ensure_prompt_parameters(self) -> None:
        if self._spec is not None:
            return
        _, model, _ = self._ensure_bundle()
        prompt_dim = int(model.hidden_dim)
        if self.prompt_mode == "fixed_full_image_box":
            self._spec = Sam3PromptSpec(
                mode=self.prompt_mode,
                trainable=False,
                prompt_init="fixed_full_image_box",
                prompt_dim=prompt_dim,
            )
            self._initial_prompt_snapshot = torch.zeros(1, 1, prompt_dim, device=self.device)
            return
        if self.prompt_mode == "learned_constant_sparse":
            param = torch.nn.Parameter(
                torch.empty(self.sparse_num_tokens, 1, prompt_dim, device=self.device)
            )
            torch.nn.init.normal_(param, mean=0.0, std=self.sparse_init_std)
            self.learned_sparse_tokens = param
            self._spec = Sam3PromptSpec(
                mode=self.prompt_mode,
                trainable=True,
                prompt_init=f"normal(0,{self.sparse_init_std})",
                prompt_num_tokens=self.sparse_num_tokens,
                prompt_dim=prompt_dim,
            )
            self._initial_prompt_snapshot = self.learned_sparse_tokens.detach().clone()
            return
        if self.prompt_mode == "fixed_full_image_box_plus_learned_delta":
            param = torch.nn.Parameter(torch.zeros(1, 1, prompt_dim, device=self.device))
            self.learned_delta = param
            self._spec = Sam3PromptSpec(
                mode=self.prompt_mode,
                trainable=True,
                prompt_init="zeros",
                prompt_num_tokens=1,
                prompt_dim=prompt_dim,
            )
            self._initial_prompt_snapshot = self.learned_delta.detach().clone()
            return
        raise ValueError(f"Unsupported prompt_mode={self.prompt_mode!r}")

    def _set_image(self, image: Image.Image) -> Dict[str, Any]:
        assert self.transform is not None
        _, model, _ = self._ensure_bundle()
        image_t = v2.functional.to_image(image).to(self.device)
        image_t = self.transform(image_t).unsqueeze(0)
        with torch.no_grad():
            backbone_out = model.backbone.forward_image(image_t)
        return {
            "original_height": int(image.height),
            "original_width": int(image.width),
            "backbone_out": backbone_out,
        }

    def _dummy_find_stage(self, batch_size: int = 1) -> Any:
        from sam3.model.data_misc import FindStage

        return FindStage(
            img_ids=torch.zeros(batch_size, device=self.device, dtype=torch.long),
            text_ids=torch.zeros(batch_size, device=self.device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

    def _apply_prompt(self, state: Dict[str, Any]) -> Dict[str, Any]:
        _, model, dummy_prompt_fn = self._ensure_bundle()
        with torch.no_grad():
            text_outputs = model.backbone.forward_text(["visual"], device=self.device)
        state["backbone_out"].update(text_outputs)
        if self.prompt_mode == "learned_constant_sparse":
            state["geometric_prompt"] = dummy_prompt_fn()
            state["prompt_override"] = "learned_constant_sparse"
            return state
        prompt = dummy_prompt_fn()
        boxes = torch.tensor([0.5, 0.5, 1.0, 1.0], device=self.device, dtype=torch.float32).view(1, 1, 4)
        labels = torch.tensor([True], device=self.device, dtype=torch.bool).view(1, 1)
        prompt.append_boxes(boxes, labels)
        state["geometric_prompt"] = prompt
        return state

    def _encode_prompt(self, state: Dict[str, Any]) -> tuple[Any, Any, Any]:
        _, model, _ = self._ensure_bundle()
        find_input = self._dummy_find_stage()
        backbone_out = state["backbone_out"]
        geometric_prompt = state["geometric_prompt"]

        txt_ids = find_input.text_ids
        txt_feats = backbone_out["language_features"][:, txt_ids]
        txt_masks = backbone_out["language_mask"][txt_ids]
        feat_tuple = model._get_img_feats(backbone_out, find_input.img_ids)
        backbone_out, img_feats, img_pos_embeds, vis_feat_sizes = feat_tuple
        geo_feats, geo_masks = model.geometry_encoder(
            geo_prompt=geometric_prompt,
            img_feats=img_feats,
            img_sizes=vis_feat_sizes,
            img_pos_embeds=img_pos_embeds,
        )

        if self.prompt_mode == "learned_constant_sparse":
            learned = self.learned_sparse_tokens
            assert learned is not None
            batch_size = int(txt_feats.shape[1])
            visual_prompt_embed = learned.expand(-1, batch_size, -1)
            visual_prompt_mask = torch.zeros(
                batch_size,
                visual_prompt_embed.shape[0],
                dtype=txt_masks.dtype,
                device=txt_masks.device,
            )
            prompt = torch.cat([txt_feats, visual_prompt_embed], dim=0)
            prompt_mask = torch.cat([txt_masks, visual_prompt_mask], dim=1)
            return prompt, prompt_mask, backbone_out

        if self.prompt_mode == "fixed_full_image_box_plus_learned_delta":
            delta = self.learned_delta
            assert delta is not None
            geo_feats = geo_feats + delta.expand(geo_feats.shape[0], geo_feats.shape[1], -1)

        prompt = torch.cat([txt_feats, geo_feats], dim=0)
        prompt_mask = torch.cat([txt_masks, geo_masks], dim=1)
        return prompt, prompt_mask, backbone_out

    def _run_forward_grounding_with_hook(self, state: Dict[str, Any]) -> torch.Tensor:
        _, model, _ = self._ensure_bundle()
        captured: list[torch.Tensor] = []

        def _capture_decoder_map(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            captured.append(output)

        handle = model.segmentation_head.pixel_decoder.register_forward_hook(_capture_decoder_map)
        try:
            prompt, prompt_mask, backbone_out = self._encode_prompt(state)
            find_input = self._dummy_find_stage()
            backbone_out, encoder_out, _ = model._run_encoder(
                backbone_out, find_input, prompt, prompt_mask
            )
            out = {
                "encoder_hidden_states": encoder_out["encoder_hidden_states"],
                "prev_encoder_out": {
                    "encoder_out": encoder_out,
                    "backbone_out": backbone_out,
                },
            }
            out, hs = model._run_decoder(
                memory=out["encoder_hidden_states"],
                pos_embed=encoder_out["pos_embed"],
                src_mask=encoder_out["padding_mask"],
                out=out,
                prompt=prompt,
                prompt_mask=prompt_mask,
                encoder_out=encoder_out,
            )
            seg_img_ids = find_input.img_ids
            model._run_segmentation_heads(
                out=out,
                backbone_out=backbone_out,
                img_ids=seg_img_ids,
                vis_feat_sizes=encoder_out["vis_feat_sizes"],
                encoder_hidden_states=out["encoder_hidden_states"],
                prompt=prompt,
                prompt_mask=prompt_mask,
                hs=hs,
            )
        finally:
            handle.remove()
        if not captured:
            raise RuntimeError("SAM3 pixel decoder hook did not capture a decoder semantic map.")
        return captured[-1]
