# Reproducibility

## Main method: MoNuSeg final_staged

This is the canonical path for the paper headline result
`0.8539 / 0.7459` on MoNuSeg-strict-512.

```bash
# Stage 1: A3 warm-up.
python scripts/train_official.py \
    --config configs/official/monuseg_strict_512.yaml \
    --variant a3_memory \
    --seed 0 \
    --output-dir checkpoints/monuseg/A3 \
    --epochs-stage1 40

# Stage 2: staged final method, warm-started from the A3 checkpoint.
python scripts/train_official.py \
    --config configs/official/monuseg_strict_512.yaml \
    --variant final_staged \
    --seed 0 \
    --init-from-checkpoint checkpoints/monuseg/A3/checkpoint.pt \
    --output-dir checkpoints/monuseg/final_staged \
    --epochs-stage2 40

# Official evaluation on the held-out test split.
python scripts/eval_official.py \
    --config configs/official/monuseg_strict_512.yaml \
    --method-config configs/official/method_staged_multiscale.yaml \
    --checkpoint checkpoints/monuseg/final_staged/checkpoint.pt \
    --output results/official_metrics/monuseg/final_staged

# Paper tables.
python scripts/make_paper_tables.py \
    --metrics-root results/official_metrics \
    --lock-file results/official_result_lock.yaml \
    --output-dir paper_assets/tables

# Comparison grids.
python scripts/make_official_visuals.py \
    --config configs/official/monuseg_strict_512.yaml \
    --checkpoint-dir checkpoints/monuseg \
    --output-dir paper_assets/qualitative/monuseg \
    --limit 14 \
    --device cuda
```

Audit-only: if you only mount the canonical final_staged checkpoint, add
`--final-staged-only` instead of silently skipping A0/A2/A3.

Expected files:

- `checkpoints/monuseg/A3/checkpoint.pt`
- `checkpoints/monuseg/final_staged/checkpoint.pt`
- `results/official_metrics/monuseg/final_staged/official_metrics.json`
- `paper_assets/qualitative/monuseg/visuals_manifest.json`

## Setup

```bash
git clone <repo> omniSAM
cd omniSAM
python -m venv .venv && source .venv/bin/activate
pip install -e .[sam]            # add `sam2` extra to pull Meta's SAM-2 package
pytest -q
```

## Datasets

- MoNuSeg: streams from `RationAI/MoNuSeg` on Hugging Face. Set
  `HF_HOME=./.cache/hf` to keep the cache local.
- GlaS: download the Warwick GlaS archive, extract under `./datasets/glas/`.
  The loader auto-resolves a flat AutoSAM-style directory inside.

## Full pipeline: MoNuSeg-strict-512

```bash
# Stage 1 (A3): 40 epochs.
python scripts/train_official.py \
    --config configs/official/monuseg_strict_512.yaml \
    --variant a3_memory \
    --seed 0 \
    --output-dir checkpoints/monuseg/stage1_seed0 \
    --epochs-stage1 40

# Stage 2 (final staged): warm-start A3, +40 epochs joint training.
python scripts/train_official.py \
    --config configs/official/monuseg_strict_512.yaml \
    --variant final_staged \
    --seed 0 \
    --init-from-checkpoint checkpoints/monuseg/stage1_seed0/checkpoint.pt \
    --output-dir checkpoints/monuseg/final_staged_seed0 \
    --epochs-stage2 40

# Official eval -> writes official_metrics.json.
python scripts/eval_official.py \
    --config configs/official/monuseg_strict_512.yaml \
    --method-config configs/official/method_staged_multiscale.yaml \
    --checkpoint checkpoints/monuseg/final_staged_seed0/checkpoint.pt \
    --output    results/official_metrics/monuseg/final_staged_seed0

# Paper tables.
python scripts/make_paper_tables.py

# Comparison grids.
python scripts/make_official_visuals.py \
    --config configs/official/monuseg_strict_512.yaml \
    --checkpoint-dir checkpoints/monuseg \
    --output-dir paper_assets/qualitative/monuseg \
    --limit 6
```

## Seeds and determinism

- All scripts accept `--seed`; defaults to `0`. The locked final-staged number
  is the mean of seeds 0, 1, 2.
- `set_seed` seeds Python, NumPy, PyTorch (CPU + CUDA).
- Frozen-SAM extraction is deterministic given the input image.

## Audit

```bash
python scripts/audit_protocol.py --config configs/official/monuseg_strict_512.yaml --check-backbone
```

## Smoke test

```bash
bash scripts/run_smoke_test.sh
```
