"""Lightweight run manifest writer."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def write_run_manifest(
    output_dir: str | Path,
    *,
    config: Dict[str, Any],
    extras: Dict[str, Any] | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "config": config,
        "extras": extras or {},
    }
    out_path = output_dir / "run_manifest.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
    return out_path
