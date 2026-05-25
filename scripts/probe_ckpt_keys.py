#!/usr/bin/env python3
"""Probe checkpoint state_dict keys for key remapper development."""
import json
import sys
from pathlib import Path

CKPTS = {
    "monuseg_a0": "/storage/nada/outputs/fixed_resize_enhancement/monuseg_A0_fpn2_d128_512_e40/checkpoint.pt",
    "monuseg_a3": "/storage/nada/outputs/fixed_resize_enhancement/monuseg_A3_fpn2fpn1memoryattn_512_m8_e40/checkpoint.pt",
    "monuseg_final_staged": "/storage/nada/outputs/fixed_resize_enhancement/monuseg_A5lrn_staged_from_A3s0_e40/checkpoint.pt",
    "glas_a0": "/storage/nada/outputs/fixed_resize_enhancement/glas_A0_fpn2_d128_224_autosamaug_e200/checkpoint.pt",
}

OUT = Path("/storage/nada/outputs/ckpt_key_probe.json")

import torch

results = {}
for name, path in CKPTS.items():
    p = Path(path)
    if not p.exists():
        results[name] = {"error": "not found"}
        continue
    sd = torch.load(p, map_location="cpu", weights_only=False)
    if "model_state_dict" in sd:
        keys = list(sd["model_state_dict"].keys())
        meta = {k: v for k, v in sd.items() if k != "model_state_dict"}
    else:
        keys = list(sd.keys())
        meta = {}
    results[name] = {"keys": keys, "meta_keys": list(meta.keys())}
    print(f"{name}: {len(keys)} keys, first 5: {keys[:5]}")

OUT.write_text(json.dumps(results, indent=2))
print(f"\nWritten to {OUT}")
