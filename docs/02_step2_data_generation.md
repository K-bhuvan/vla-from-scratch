# Step 2: Synthetic Data Generation

## Goal
Create a small, fully local, fully open dataset that mimics robotics demonstration data without depending on a heavy simulator yet.

## What was created
- `data/generate_sim_data.py`
- `configs/data_gen.yaml`
- generated HDF5 episodes under `data/raw/sim_demos/`

## Tasks implemented
- `reach_target`
- `pick_object` (primitive)
- `place_object` (primitive)
- `pick_and_place_object` (composed; object type sampled as ball/cube/cylinder)

## Why this step matters
All later stages depend on a stable data format. Before training a VLA, you need a consistent schema for:
- observations
- robot state
- actions
- episode metadata

This is exactly how real robotics stacks are built: define the data contract first, then train models against that contract.

## Data schema
Each HDF5 episode contains:
- `obs_image`: `(T, 84, 84, 3)` RGB frames with shape `(T, H, W, C)` using HWC layout (height, width, channels), where `T` is time and `C = 3` for RGB.
- `state`: `(T, 7)` robot state
- `action`: `(T, 7)` delta action
- `done`: `(T,)` episode termination flag

Attributes stored per file:
- `task_name`
- `instruction`
- `success`
- `episode_len`
- `object_type` (for `pick_object`, `place_object`, `pick_and_place_object`)

## Robot representation
State vector:
- `[x, y, z, roll, pitch, yaw, gripper]`

Action vector:
- `[dx, dy, dz, d_roll, d_pitch, d_yaw, d_gripper]`

We store delta actions because later the policy should learn incremental control, not absolute teleport targets.

## Oracle policies
Scripted experts generate demonstrations:
- `oracle_reach_target`: proportional controller toward a target
- `oracle_pick_object`: four-phase routine
  - move above object
  - descend
  - close gripper
  - lift
- `oracle_place_object`: place-only routine
  - move above box
  - descend
  - open gripper
  - retreat
- `oracle_pick_and_place_object`: multi-phase pick-and-drop-off routine
  - approach object
  - grasp
  - lift
  - move above box
  - descend and release
  - retreat

## Rendering approach
The scene is rendered as a simple top-down RGB image with Pillow:
- blue robot drawing = base + links + end-effector + gripper claws
- green circle = target (`reach_target`)
- orange/red/purple object = ball/cube/cylinder (`pick_object`, `place_object`, `pick_and_place_object`)
- dark red square = drop-off box (`place_object`, `pick_and_place_object`)

This is intentionally simple. The purpose is to learn the full pipeline first, not photorealism.

## How to run
Quick test:
```bash
conda activate vla
cd <repo-root>
python data/generate_sim_data.py --num-episodes 20 --tasks pick_and_place_object --inspect
```

Full data generation:
```bash
python data/generate_sim_data.py
```

## What you learned
- how robot data is structured episode by episode
- why primitive + composed task mixtures improve generalization
- why behavior cloning relies on clean demonstrations
- why HDF5 is a common robotics dataset format
- how multi-stage tasks contain implicit phase information

## Output of this step
A local synthetic dataset that is sufficient for pretraining and SFT experiments in the next stages.
