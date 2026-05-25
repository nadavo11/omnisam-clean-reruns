# Method: Staged Multiscale Frozen-SAM3 Readout

## Inputs and notation

Let `I` be an input image, resized to the protocol resolution before being fed
to the frozen SAM image encoder. The encoder produces a 3-level pyramid:

- `F2 = fpn_2` (coarsest, smallest spatial resolution, deepest channels)
- `F1 = fpn_1` (middle)
- `F0 = fpn_0` (finest)

The SAM prompt encoder and mask decoder are NEVER invoked. All SAM parameters
have `requires_grad = False`; this is enforced by
`frozen_sam_readout.sam.freeze_utils.assert_sam_frozen`.

## Stage 1 — A3 head

The A3 head produces a coarse logit map `L_A3` at the `F1` resolution.

```
Z2  = proj_2(F2)                                  # 1x1 conv, channels -> D
Z2' = MemoryAttention(Z2, M=8)                    # learned tokens cross-attend
C   = Conv1(Conv0(Z2'))                           # 2 x ConvBNReLU
Lc  = head_coarse(C)                              # 1x1 -> 1 channel
Z1  = proj_1(F1)                                  # 1x1 conv, channels -> D
Lr  = head_refine(Fuse(Z1, up(Lc)))               # 1x1 -> 1 channel
L_A3 = Lr                                         # final coarse-resolution logits
```

- `M = 8` memory tokens.
- `D = 128` (`decoder_dim`, `projection_dim`).
- Trained for 40 epochs (MoNuSeg) or 200 epochs (GlaS) with AdamW, lr `1e-4`,
  weight-decay `1e-4`, batch size 1.

## Stage 2 — Learned-alpha F0 residual

```
Z0  = proj_0(F0)
R0  = head_final( Fuse(Z0, up(L_A3)) )            # head_final is zero-init
alpha       = sigmoid(raw_alpha)
raw_alpha_0 = log(0.05 / 0.95)                    # so alpha ≈ 0.05 at init
L_final     = up(L_A3) + alpha * R0
```

The final 1×1 convolution inside the F0 residual head is zero-initialized so
`R0 = 0` at the start of stage-2 training, i.e. `L_final ≈ up(L_A3)`. This
keeps stage-1 quality intact and lets the F0 branch gradually contribute as
training progresses. Trained for an additional **+40 epochs**, jointly
updating A3 and the F0 branch (do NOT freeze A3 by default).

## Prediction

```
Pred = sigmoid(L_final) > 0.5
```

Pred is bilinearly upsampled to the ground-truth raster before thresholding
so the headline metrics (`direct_foreground_dice`,
`direct_foreground_iou`) are computed on the GT grid.

## Ablation variants

| ID | Description |
|---|---|
| A0 | Coarse readout on `fpn_2` alone. |
| A2 | A0 + `fpn_1` residual refine block. |
| A3 | A2 + memory-attention tokens on the projected `fpn_2`. |
| A5 all-at-once | Same architecture as final, but trained jointly from scratch (no stage-1 warm-start). |
| Final staged | The official method. |
| Freeze-A3 control | Stage-2 with A3 frozen. Underperforms the joint baseline. |
