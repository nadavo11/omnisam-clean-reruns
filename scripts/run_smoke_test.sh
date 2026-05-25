#!/usr/bin/env bash
# Quick smoke-test runner: validates configs, runs pytest, and (optionally)
# launches a 1-sample eval against a tiny mock checkpoint.

set -euo pipefail

cd "$(dirname "$0")/.."

echo "== pytest =="
python -m pytest -q tests

echo "== audit_protocol on MoNuSeg strict 512 =="
python scripts/audit_protocol.py --config configs/official/monuseg_strict_512.yaml

echo "== audit_protocol on GlaS fixed 224 =="
python scripts/audit_protocol.py --config configs/official/glas_fixed_224.yaml

echo "Smoke test complete."
