# Baby Step 8: Stage B SFT - Behavior Cloning

## Goal

Now that Step 7 aligned vision and language, Step 8 teaches the full policy to imitate expert actions.

Input at each timestep:
- image_t
- instruction

Target:
- expert action_t = [dx, dy, dz, d_roll, d_pitch, d_yaw, d_gripper]

## Model used

Step 8 trains the full 4-module pipeline:
- VisionEncoder
- LanguageEncoder
- VisionLanguageFusion
- ActionHead

The vision and language encoders can be warm-started from Step 7 using:
- outputs/pretrain/best.pt

Fusion and action head are initialized fresh and trained jointly.

## Dataset construction

Unlike Step 7, which sampled one random frame per episode, Step 8 uses timestep-level samples.

That means one episode with T timesteps contributes T supervised training examples:
- (image_0, instruction) -> action_0
- (image_1, instruction) -> action_1
- ...
- (image_T-1, instruction) -> action_T-1

This is the correct setup for behavior cloning because actions are defined per timestep.

## Loss

Default loss is Huber regression:
- more stable than pure MSE when a few target actions are noisy or large
- still behaves like MSE near zero error

You can switch to MSE in configs/sft.yaml if needed.

## Important MVP note

The current Step 8 trainer uses image + instruction only, because the model architecture built so far does not yet include a state encoder.

The HDF5 demos do contain robot state, so a later upgrade can add state conditioning. For this MVP we keep the architecture consistent with the modules already implemented.

## Run

Full run:

```bash
python -m src.train.sft --config configs/sft.yaml
```

Note: smoke configs were removed from the publishable repo to keep it clean. Use `configs/sft.yaml` for reproducible runs.

## Outputs

Checkpoints are written to:
- outputs/sft/best.pt
- outputs/sft/last.pt

These will be the starting point for Step 9 and Step 10.
