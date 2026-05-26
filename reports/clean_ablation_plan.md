# Clean Ablation Plan

## Policy

Every run is classified as:

- **E0**: smoke / fit test, train-only, no paper claim
- **E1**: validation ablation, held-out validation only
- **E2**: official test / paper result, one locked run per candidate

No result enters the paper unless it is E2 or explicitly labeled validation-only.

## Immediate priorities

1. Rebuild the evidence stack on a clean split discipline.
2. Treat all existing MoNuSeg ablations as invalidated evidence.
3. Preserve checkpoints and logs as candidates and diagnostics only.
4. Keep GlaS official metrics as the only currently green headline evidence.

## Minimal clean rerun matrix

Start with the smallest decision matrix:

| Question | Variants | Purpose |
| --- | --- | --- |
| Does frozen SAM evidence help at all? | simple CNN/UNet head vs frozen SAM feature head | establish the premise |
| Which feature level matters? | `fpn_2` only / `fpn_1` only / `fpn_2+fpn_1` | isolate representation source |
| Does memory help generalization? | no memory / memory `M=4` / memory `M=8` | separate signal from memorization |
| Does staged training help? | single-stage / staged from best simple seed | test optimization path, not architecture hype |
| Does repair/refinement help? | core-only / refine-only / final | check overlap-vs-consistency tradeoff |

## Required artifacts per run

Each future run must emit:

- `config.yaml`
- `split_manifest.json`
- `checkpoint_sha256.txt`
- `eval_metrics_val.json`
- `eval_metrics_test.json`
- `visual_sample_ids.txt`
- `visuals/`
- `wandb_run_url.txt` when W&B is available

## Visual policy

- Fixed sample IDs only
- Never cherry-pick by metric
- Generate both validation and test visuals once the model is locked

## First rerun order

1. `A0` anchor, 3 seeds, validation + test
2. `A2` multiscale refine, 3 seeds, validation + test
3. `A3` memory attention, 3 seeds, validation + test
4. `A5` staged/refine only if the parent variants beat `A0` on validation

## Success criteria

A module only counts as useful if it:

- improves held-out Dice/IoU over its direct parent
- does not collapse visuals
- improves at least 2 of 3 seeds or shows a convincing paired improvement
- does not rely on test-set-selected thresholding
- has an explainable failure mode

