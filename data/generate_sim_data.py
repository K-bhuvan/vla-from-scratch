"""
data/generate_sim_data.py  —  Baby Step 2: Synthetic Trajectory Data Generator

────────────────────────────────────────────────────────────────────────────────
WHAT THIS SCRIPT DOES
────────────────────────────────────────────────────────────────────────────────
Generates synthetic robot demonstration trajectories and saves them as HDF5
files.  No real simulator is needed — we use NumPy to compute robot motion and
Pillow to render simple top-down scene images.

WHY WE BUILD THE DATA PIPELINE BEFORE ANY MODEL CODE
────────────────────────────────────────────────────────────────────────────────
Every training stage (pre-training, SFT, post-training) reads the SAME data
format.  If we nail the schema now, the training code never has to guess.
This is exactly how real robotics teams work: data engineers define the schema
first, then model engineers write code that consumes it.

WHAT A "TRAJECTORY" IS
────────────────────────────────────────────────────────────────────────────────
One trajectory = one episode = one complete task attempt.

At every timestep t the robot records:
  obs_image[t]  — what the camera sees  (H × W × 3 pixels)
  state[t]      — its own joint/EEF readings (7 numbers)
  action[t]     — the command it sends    (7 numbers)

After T timesteps the episode ends and we record whether it succeeded.

STATE VECTOR  [x, y, z, roll, pitch, yaw, gripper]
  x, y, z       — end-effector Cartesian position in metres
  roll,pitch,yaw— end-effector orientation in radians (kept at 0 here)
  gripper       — openness: 1.0 = fully open, 0.0 = fully closed

ACTION VECTOR  [dx, dy, dz, d_roll, d_pitch, d_yaw, d_gripper]
  Delta (incremental) commands applied ON TOP of the current state.
  Delta actions are standard because:
    • They are naturally bounded (small steps → safer hardware commands)
    • The policy learns incremental corrections, not absolute goals
    • Easier to normalize during training

WHY HDF5 FORMAT?
────────────────────────────────────────────────────────────────────────────────
HDF5 is the de-facto format for large robotics datasets (RoboMimic, BridgeData,
Open-X Embodiment).  Benefits:
  • Stores arrays of ANY dtype/shape with compression
  • Fast random access by index — critical for shuffled training batches
  • Metadata (task name, instruction, success) stored as HDF5 attributes

TASKS IMPLEMENTED
────────────────────────────────────────────────────────────────────────────────
  reach_target — move end-effector to a random 3-D target position
    pick_object  — approach an object, grasp, and lift
    place_object — move grasped object to box and release
    pick_and_place_object — full sequence: pick + place (drop-off)

RUNNING
────────────────────────────────────────────────────────────────────────────────
  conda activate vla
  python data/generate_sim_data.py                       # defaults: 500 ep/task
  python data/generate_sim_data.py --num-episodes 100    # quick test run
  python data/generate_sim_data.py --inspect             # inspect a saved file
"""

import argparse
import math
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# DATA SCHEMA CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

STATE_DIM  = 10  # [x, y, z, roll, pitch, yaw, gripper, goal_dx, goal_dy, goal_dz]
ACTION_DIM = 7   # [dx, dy, dz, d_roll, d_pitch, d_yaw, d_gripper]

# Robot's reachable workspace in metres (3-D bounding box).
# The model will need to reason about positions within this volume.
WORKSPACE = {
    "x": (-0.5,  0.5),
    "y": (-0.5,  0.5),
    "z": ( 0.0,  0.5),
}

# Image size.  84×84 is standard for many robot-learning benchmarks (e.g.
# DeepMind Control Suite, RLBench low-res variant).  Keeps data files small.
IMG_H, IMG_W = 84, 84

TASKS = ["reach_target", "pick_object", "place_object", "pick_and_place_object"]

# Multiple phrasings per task — the model must generalise across all of them.
# This is how real teams build instruction-following diversity.
INSTRUCTIONS = {
    "reach_target": [
        "move to the green target",
        "navigate the arm to the green marker",
        "reach the target position",
        "go to the green circle",
        "bring the end-effector to the green dot",
    ],
    "pick_object": [
        "pick up the {object_name}",
        "grasp the {object_name}",
        "lift the {object_name}",
        "grab the {object_name}",
        "pick the {object_name} off the table",
    ],
    "place_object": [
        "place the {object_name} in the box",
        "move the {object_name} to the drop box",
        "drop off the {object_name} at the box",
        "carry the {object_name} and release it in the box",
        "put the {object_name} into the box",
    ],
    "pick_and_place_object": [
        "pickup the {object_name} & drop-off at box",
        "pick up the {object_name} and place it in the box",
        "grab the {object_name} and drop it into the box",
        "move the {object_name} to the drop box and release it",
        "pick the {object_name} and place it in the drop-off box",
    ],
}

OBJECT_TYPES = ["ball", "cube", "cylinder"]

OBJECT_RENDER_STYLE = {
    "ball": {
        "fill": (230, 140, 35),
        "edge": (180, 95, 10),
    },
    "cube": {
        "fill": (220, 85, 85),
        "edge": (160, 35, 35),
    },
    "cylinder": {
        "fill": (195, 120, 220),
        "edge": (120, 65, 150),
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# ORACLE TRAJECTORIES
#
# An "oracle" (or "scripted expert") is a hard-coded policy that always does
# the right thing.  In the real world you get these from:
#   • Human teleoperation recordings
#   • Motion-planning algorithms (e.g. RRT, CHOMP)
#   • Reinforcement learning after it has converged
#
# We will train our VLA to IMITATE these oracle trajectories — that is the
# core idea of Behaviour Cloning (Baby Step 8).
# ─────────────────────────────────────────────────────────────────────────────

def oracle_reach_target(
    start_xyz: np.ndarray,
    target_xyz: np.ndarray,
    max_steps: int = 40,
    success_thresh: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Scripted expert for the reach_target task.

    Strategy: proportional control — each step moves a fixed fraction of the
    remaining distance toward the target (like a simple P-controller).

    Parameters
    ----------
    start_xyz       : initial end-effector (x, y, z) position
    target_xyz      : goal (x, y, z) position
    max_steps       : episode length cap
    success_thresh  : distance in metres counted as "reached"

    Returns
    -------
    states  : (T, STATE_DIM) float32
    actions : (T, ACTION_DIM) float32
    success : bool
    """
    states  = []
    actions = []

    state = np.zeros(STATE_DIM, dtype=np.float32)
    state[:3] = start_xyz.copy()
    state[3:6] = 0.0   # orientation kept at zero (simplified)
    state[6]   = 1.0   # gripper fully open

    # gain: fraction of remaining distance moved per step.
    # At 0.25 the controller reaches within 5 cm of a 1-m gap in ~14 steps.
    gain = 0.25

    for _ in range(max_steps):
        state[7:10] = target_xyz - state[:3]   # goal delta: vector to target
        states.append(state.copy())

        delta_xyz = target_xyz - state[:3]
        dist = float(np.linalg.norm(delta_xyz))

        action = np.zeros(ACTION_DIM, dtype=np.float32)
        if dist >= success_thresh:
            action[:3] = gain * delta_xyz   # proportional step toward target

        actions.append(action.copy())

        # Advance state by applying action, then clip to workspace bounds
        state = state.copy()
        state[:3] += action[:3]
        state[:3] = _clip_to_workspace(state[:3])

        if float(np.linalg.norm(target_xyz - state[:3])) < success_thresh:
            break

    final_dist = float(np.linalg.norm(target_xyz - state[:3]))
    success = final_dist < success_thresh

    return (
        np.array(states,  dtype=np.float32),
        np.array(actions, dtype=np.float32),
        success,
    )


def oracle_pick_object(
    start_xyz: np.ndarray,
    object_xyz: np.ndarray,
    max_steps: int = 60,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Scripted expert for the pick_object task.

    The pick sequence has four phases:
    Phase 1 — move ABOVE the object (pre-grasp approach, z = object_z + 0.15 m)
    Phase 2 — DESCEND to the object height
      Phase 3 — CLOSE the gripper
      Phase 4 — LIFT upward (z = cube_z + 0.25 m)

    LEARNING NOTE — why phases matter:
      Multi-phase tasks are common in manipulation.  The model must learn to
      infer the current phase from images + history and execute the right
      sub-motion.  Demonstration data implicitly encodes this phase structure —
      which is one reason behaviour cloning outperforms naive RL on long-horizon
      tasks: the demonstrations carry temporal context for free.
    """
    states  = []
    actions = []

    state = np.zeros(STATE_DIM, dtype=np.float32)
    state[:3] = start_xyz.copy()
    state[6]  = 1.0  # gripper open

    gain   = 0.30
    thresh = 0.04    # metres

    # Phase waypoints
    above_object = np.array([object_xyz[0], object_xyz[1], object_xyz[2] + 0.15], dtype=np.float32)
    at_object    = np.array([object_xyz[0], object_xyz[1], object_xyz[2]       ], dtype=np.float32)
    lift_target = np.array([object_xyz[0], object_xyz[1], object_xyz[2] + 0.25], dtype=np.float32)

    phase = 1  # start at phase 1

    for _ in range(max_steps):
        # goal_delta: vector from EEF to current sub-goal waypoint
        if phase == 1:
            _goal = above_object
        elif phase == 2:
            _goal = at_object
        elif phase == 3:
            _goal = at_object
        else:  # phase == 4
            _goal = lift_target
        state[7:10] = _goal - state[:3]
        states.append(state.copy())
        action = np.zeros(ACTION_DIM, dtype=np.float32)

        if phase == 1:
            # Move above object
            delta = above_object - state[:3]
            if np.linalg.norm(delta) < thresh:
                phase = 2
            else:
                action[:3] = gain * delta

        elif phase == 2:
            # Descend to object
            delta = at_object - state[:3]
            if np.linalg.norm(delta) < thresh:
                phase = 3
            else:
                action[:3] = gain * delta

        elif phase == 3:
            # Close gripper
            action[6] = -0.4  # negative = close
            if state[6] <= 0.05:
                phase = 4

        elif phase == 4:
            # Lift up
            delta = lift_target - state[:3]
            action[:3] = gain * delta

        actions.append(action.copy())

        # Apply action
        state = state.copy()
        state[:3] += action[:3]
        state[6]   = float(np.clip(state[6] + action[6], 0.0, 1.0))
        state[:3]  = _clip_to_workspace(state[:3])

        if state[6] < 0.15 and state[2] > object_xyz[2] + 0.08:
            break

    # Success: gripper closed AND EEF is lifted above the object
    success = bool(state[6] < 0.15 and state[2] > object_xyz[2] + 0.08)

    return (
        np.array(states,  dtype=np.float32),
        np.array(actions, dtype=np.float32),
        success,
    )


def oracle_pick_and_place_object(
    start_xyz: np.ndarray,
    object_xyz: np.ndarray,
    box_xyz: np.ndarray,
    max_steps: int = 70,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Scripted expert for the pick_and_place_object task.

    Sequence (8 phases):
      1. move above ball
      2. descend to ball
      3. close gripper (grasp)
      4. lift ball
      5. move above box
      6. descend to box drop height
      7. open gripper (release)
      8. retreat up

    This adds a proper drop-off behavior to demonstration data.
    """
    states = []
    actions = []

    state = np.zeros(STATE_DIM, dtype=np.float32)
    state[:3] = start_xyz.copy()
    state[6] = 1.0  # gripper open

    gain = 0.30
    thresh = 0.04

    above_object = np.array([object_xyz[0], object_xyz[1], object_xyz[2] + 0.15], dtype=np.float32)
    at_object = np.array([object_xyz[0], object_xyz[1], object_xyz[2]], dtype=np.float32)
    lift_after_pick = np.array([object_xyz[0], object_xyz[1], object_xyz[2] + 0.25], dtype=np.float32)
    above_box = np.array([box_xyz[0], box_xyz[1], box_xyz[2] + 0.18], dtype=np.float32)
    at_box_drop = np.array([box_xyz[0], box_xyz[1], box_xyz[2] + 0.08], dtype=np.float32)
    retreat = np.array([box_xyz[0], box_xyz[1], box_xyz[2] + 0.22], dtype=np.float32)

    phase = 1
    released_near_box = False

    for _ in range(max_steps):
        # goal_delta: vector from EEF to current sub-goal waypoint
        if phase == 1:
            _goal = above_object
        elif phase == 2:
            _goal = at_object
        elif phase == 3:
            _goal = at_object
        elif phase == 4:
            _goal = lift_after_pick
        elif phase == 5:
            _goal = above_box
        elif phase == 6:
            _goal = at_box_drop
        elif phase == 7:
            _goal = at_box_drop
        else:  # phase == 8
            _goal = retreat
        state[7:10] = _goal - state[:3]
        states.append(state.copy())
        action = np.zeros(ACTION_DIM, dtype=np.float32)

        if phase == 1:
            delta = above_object - state[:3]
            if np.linalg.norm(delta) < thresh:
                phase = 2
            else:
                action[:3] = gain * delta

        elif phase == 2:
            delta = at_object - state[:3]
            if np.linalg.norm(delta) < thresh:
                phase = 3
            else:
                action[:3] = gain * delta

        elif phase == 3:
            action[6] = -0.4  # close
            if state[6] <= 0.05:
                phase = 4

        elif phase == 4:
            delta = lift_after_pick - state[:3]
            if np.linalg.norm(delta) < thresh:
                phase = 5
            else:
                action[:3] = gain * delta

        elif phase == 5:
            delta = above_box - state[:3]
            if np.linalg.norm(delta) < thresh:
                phase = 6
            else:
                action[:3] = gain * delta

        elif phase == 6:
            delta = at_box_drop - state[:3]
            if np.linalg.norm(delta) < thresh:
                phase = 7
            else:
                action[:3] = gain * delta

        elif phase == 7:
            action[6] = +0.4  # open and release
            # Keep track of release quality near the drop-off box.
            xy_dist = float(np.linalg.norm(state[:2] - box_xyz[:2]))
            if xy_dist < 0.07:
                released_near_box = True
            if state[6] >= 0.95:
                phase = 8

        elif phase == 8:
            delta = retreat - state[:3]
            action[:3] = gain * delta

        actions.append(action.copy())

        state = state.copy()
        state[:3] += action[:3]
        state[6] = float(np.clip(state[6] + action[6], 0.0, 1.0))
        state[:3] = _clip_to_workspace(state[:3])

        if released_near_box and state[6] > 0.9 and state[2] > box_xyz[2] + 0.10:
            break

    success = bool(
        released_near_box
        and state[6] > 0.9
        and state[2] > box_xyz[2] + 0.10
    )

    return (
        np.array(states, dtype=np.float32),
        np.array(actions, dtype=np.float32),
        success,
    )


def oracle_place_object(
    start_xyz: np.ndarray,
    box_xyz: np.ndarray,
    max_steps: int = 55,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Scripted expert for place_object.

    Assumption: the object is already grasped at episode start (gripper closed).
    Sequence: move above box -> descend -> open gripper -> retreat.
    """
    states = []
    actions = []

    state = np.zeros(STATE_DIM, dtype=np.float32)
    state[:3] = start_xyz.copy()
    state[6] = 0.0  # closed: object already in hand

    gain = 0.30
    thresh = 0.04

    above_box = np.array([box_xyz[0], box_xyz[1], box_xyz[2] + 0.18], dtype=np.float32)
    at_box_drop = np.array([box_xyz[0], box_xyz[1], box_xyz[2] + 0.08], dtype=np.float32)
    retreat = np.array([box_xyz[0], box_xyz[1], box_xyz[2] + 0.22], dtype=np.float32)

    phase = 1
    released_near_box = False

    for _ in range(max_steps):
        # goal_delta: vector from EEF to current sub-goal waypoint
        if phase == 1:
            _goal = above_box
        elif phase == 2:
            _goal = at_box_drop
        elif phase == 3:
            _goal = at_box_drop
        else:  # phase == 4
            _goal = retreat
        state[7:10] = _goal - state[:3]
        states.append(state.copy())
        action = np.zeros(ACTION_DIM, dtype=np.float32)

        if phase == 1:
            delta = above_box - state[:3]
            if np.linalg.norm(delta) < thresh:
                phase = 2
            else:
                action[:3] = gain * delta

        elif phase == 2:
            delta = at_box_drop - state[:3]
            if np.linalg.norm(delta) < thresh:
                phase = 3
            else:
                action[:3] = gain * delta

        elif phase == 3:
            action[6] = +0.4  # open and release
            xy_dist = float(np.linalg.norm(state[:2] - box_xyz[:2]))
            if xy_dist < 0.07:
                released_near_box = True
            if state[6] >= 0.95:
                phase = 4

        elif phase == 4:
            delta = retreat - state[:3]
            action[:3] = gain * delta

        actions.append(action.copy())

        state = state.copy()
        state[:3] += action[:3]
        state[6] = float(np.clip(state[6] + action[6], 0.0, 1.0))
        state[:3] = _clip_to_workspace(state[:3])

        if released_near_box and state[6] > 0.9 and state[2] > box_xyz[2] + 0.10:
            break

    success = bool(released_near_box and state[6] > 0.9 and state[2] > box_xyz[2] + 0.10)

    return (
        np.array(states, dtype=np.float32),
        np.array(actions, dtype=np.float32),
        success,
    )


def _clip_to_workspace(xyz: np.ndarray) -> np.ndarray:
    """Clip (x, y, z) to the robot's reachable workspace bounds."""
    return np.array([
        float(np.clip(xyz[0], *WORKSPACE["x"])),
        float(np.clip(xyz[1], *WORKSPACE["y"])),
        float(np.clip(xyz[2], *WORKSPACE["z"])),
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE RENDERING
#
# Produces a synthetic top-down (bird's-eye) RGB view of the scene.
# In a real setup this would come from an onboard robot camera.
# We use Pillow here so there is NO simulator dependency during early steps.
#
# What the image encodes:
#   • Blue marker   — robot end-effector position (ring + center + heading tick;
#                     size still encodes gripper state)
#   • Green circle  — target location (reach_target task)
#   • Object shape  — object location (pick_object / place_object / pick_and_place_object)
#   • Ball/cube/cylinder — object to pick and place (pick_and_place_object)
#   • Dark red box       — drop-off location (pick_and_place_object)
# ─────────────────────────────────────────────────────────────────────────────

_COLOR_BG     = (230, 230, 230)   # light grey background
_COLOR_TABLE  = (180, 150, 100)   # tan table surface
_COLOR_EEF    = ( 60, 100, 220)   # blue  — end-effector
_COLOR_EEF_HALO = (245, 245, 255) # bright halo for contrast on table background
_COLOR_EEF_CORE = ( 15,  35, 110) # dark center dot for easy localization
_COLOR_ARM_LINK = ( 90, 110, 150) # visible arm links
_COLOR_ARM_EDGE = ( 45,  60,  95) # arm outlines/joints
_COLOR_TARGET = ( 50, 200,  80)   # green — reach target
_COLOR_BOX    = (120,  35,  35)   # dark red — drop-off box


def _world_to_pixel(x: float, y: float) -> tuple[int, int]:
    """
    Map world (x, y) in metres to image pixel (px, py).

    World x ∈ [-0.5, 0.5]  →  image column  ∈ [0, IMG_W]
    World y ∈ [-0.5, 0.5]  →  image row     ∈ [IMG_H, 0]   (y-axis flipped)
    """
    px = int((x - WORKSPACE["x"][0]) / (WORKSPACE["x"][1] - WORKSPACE["x"][0]) * IMG_W)
    py = int((1.0 - (y - WORKSPACE["y"][0]) / (WORKSPACE["y"][1] - WORKSPACE["y"][0])) * IMG_H)
    px = max(0, min(IMG_W - 1, px))
    py = max(0, min(IMG_H - 1, py))
    return px, py


def render_frame(
    state: np.ndarray,
    task: str,
    target_xyz: Optional[np.ndarray] = None,
    object_xyz: Optional[np.ndarray] = None,
    object_type: Optional[str] = None,
    box_xyz:    Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Render one timestep as an (IMG_H, IMG_W, 3) uint8 RGB array.

    The image is a top-down projection — we ignore z (height) for the 2-D
    view.  This is a strong simplification; real systems use perspective
    cameras, depth sensors, or multiple views.  For learning the pipeline it
    is perfectly adequate.

        Gripper openness is encoded in the end-effector marker radius:
            large marker → gripper open
            small marker → gripper closed
    This gives the model a visual cue it can learn to associate with
    grasp actions — an important concept in visual robot learning.

        Human-readability note:
            The marker is intentionally high-contrast (halo + blue ring + dark core)
            so it is obvious at a glance in tiny 84×84 images.
    """
    img  = Image.new("RGB", (IMG_W, IMG_H), _COLOR_BG)
    draw = ImageDraw.Draw(img)

    # Table surface (slightly inset)
    m = 4
    draw.rectangle([m, m, IMG_W - m, IMG_H - m], fill=_COLOR_TABLE)

    # ── Task-specific objects ─────────────────────────────────────────────
    if task == "reach_target" and target_xyz is not None:
        tx, ty = _world_to_pixel(target_xyz[0], target_xyz[1])
        r = 5
        # Filled circle with a cross marker inside
        draw.ellipse([tx-r, ty-r, tx+r, ty+r], fill=_COLOR_TARGET, outline=(20, 140, 40), width=1)
        draw.line([tx-r, ty, tx+r, ty], fill=(20, 140, 40), width=1)
        draw.line([tx, ty-r, tx, ty+r], fill=(20, 140, 40), width=1)

    if task in ("pick_object", "place_object", "pick_and_place_object") and object_xyz is not None:
        ox, oy = _world_to_pixel(object_xyz[0], object_xyz[1])
        style = OBJECT_RENDER_STYLE.get(str(object_type), OBJECT_RENDER_STYLE["ball"])

        if object_type == "cube":
            r = 5
            draw.rectangle([ox-r, oy-r, ox+r, oy+r], fill=style["fill"], outline=style["edge"], width=1)
        elif object_type == "cylinder":
            rx, ry = 5, 3
            draw.ellipse([ox-rx, oy-ry-2, ox+rx, oy+ry-2], fill=style["fill"], outline=style["edge"], width=1)
            draw.rectangle([ox-rx, oy-ry-2, ox+rx, oy+ry+3], fill=style["fill"], outline=style["edge"], width=1)
            draw.ellipse([ox-rx, oy+1, ox+rx, oy+ry+5], fill=style["fill"], outline=style["edge"], width=1)
        else:
            r = 5
            draw.ellipse([ox-r, oy-r, ox+r, oy+r], fill=style["fill"], outline=style["edge"], width=1)

        if box_xyz is not None:
            qx, qy = _world_to_pixel(box_xyz[0], box_xyz[1])
            qr = 8
            # Box drawn as nested square to make the drop zone obvious.
            draw.rectangle([qx-qr, qy-qr, qx+qr, qy+qr], fill=(170, 130, 95), outline=_COLOR_BOX, width=2)
            draw.rectangle([qx-qr+3, qy-qr+3, qx+qr-3, qy+qr-3], outline=_COLOR_BOX, width=1)

    # ── Stylized robot arm (base + 2 links + joints) ─────────────────────
    # This is a visual aid only.  We infer link geometry from EEF position
    # so images are more intuitive for humans while keeping the data schema
    # and controller exactly the same.
    base_x, base_y = IMG_W // 2, IMG_H - 7

    # ── Robot end-effector ────────────────────────────────────────────────
    ex, ey  = _world_to_pixel(state[0], state[1])
    gripper = float(state[6])                  # 0 = closed, 1 = open
    eef_r   = max(2, int(3 + gripper * 4))     # radius encodes openness

    # Infer an elbow point with a gentle bend so the arm silhouette is clear.
    dx = ex - base_x
    dy = ey - base_y
    dist = max(1.0, math.hypot(dx, dy))
    mid_x = 0.5 * (base_x + ex)
    mid_y = 0.5 * (base_y + ey)

    # Unit vector perpendicular to base->eef
    nx = -dy / dist
    ny = dx / dist

    bend_sign = 1.0 if ex >= base_x else -1.0
    bend = min(12.0, max(5.0, 0.28 * dist))
    elbow_x = int(round(mid_x + bend_sign * bend * nx))
    elbow_y = int(round(mid_y + bend_sign * bend * ny))
    elbow_x = max(2, min(IMG_W - 3, elbow_x))
    elbow_y = max(2, min(IMG_H - 3, elbow_y))

    # Draw links with edge + fill strokes for better contrast.
    draw.line([base_x, base_y, elbow_x, elbow_y], fill=_COLOR_ARM_EDGE, width=6)
    draw.line([base_x, base_y, elbow_x, elbow_y], fill=_COLOR_ARM_LINK, width=4)
    draw.line([elbow_x, elbow_y, ex, ey], fill=_COLOR_ARM_EDGE, width=6)
    draw.line([elbow_x, elbow_y, ex, ey], fill=_COLOR_ARM_LINK, width=4)

    # Joints
    draw.ellipse([base_x-4, base_y-4, base_x+4, base_y+4], fill=_COLOR_ARM_EDGE)
    draw.ellipse([elbow_x-3, elbow_y-3, elbow_x+3, elbow_y+3], fill=_COLOR_ARM_EDGE)

    # 1) White-ish outer halo: makes the marker visible on similar backgrounds.
    halo_r = eef_r + 2
    draw.ellipse([ex-halo_r, ey-halo_r, ex+halo_r, ey+halo_r],
                 fill=_COLOR_EEF_HALO, outline=(180, 180, 210), width=1)

    # 2) Blue filled ring/body: still the primary identity color for EEF.
    draw.ellipse([ex-eef_r, ey-eef_r, ex+eef_r, ey+eef_r],
                 fill=_COLOR_EEF, outline=(20, 60, 180), width=1)

    # 3) Dark center dot: precise center location is unambiguous.
    core_r = 1 if eef_r <= 3 else 2
    draw.ellipse([ex-core_r, ey-core_r, ex+core_r, ey+core_r],
                 fill=_COLOR_EEF_CORE)

    # 4) Gripper claws using yaw + openness so "hand" is visually obvious.
    yaw = float(state[5])
    fx = math.cos(yaw)
    fy = -math.sin(yaw)  # image y-axis is inverted
    px = -fy
    py = fx

    jaw_len = eef_r + 4
    jaw_sep = 1.5 + 2.5 * gripper

    palm_x = ex - 1.0 * fx
    palm_y = ey - 1.0 * fy

    # Upper jaw
    ux0 = int(round(palm_x + jaw_sep * px))
    uy0 = int(round(palm_y + jaw_sep * py))
    ux1 = int(round(ux0 + jaw_len * fx))
    uy1 = int(round(uy0 + jaw_len * fy))

    # Lower jaw
    lx0 = int(round(palm_x - jaw_sep * px))
    ly0 = int(round(palm_y - jaw_sep * py))
    lx1 = int(round(lx0 + jaw_len * fx))
    ly1 = int(round(ly0 + jaw_len * fy))

    draw.line([ux0, uy0, ux1, uy1], fill=_COLOR_EEF_CORE, width=2)
    draw.line([lx0, ly0, lx1, ly1], fill=_COLOR_EEF_CORE, width=2)

    return np.array(img, dtype=np.uint8)  # (H, W, 3)


# ─────────────────────────────────────────────────────────────────────────────
# HDF5 EPISODE WRITER
# ─────────────────────────────────────────────────────────────────────────────

def save_episode(
    filepath: Path,
    frames:   np.ndarray,   # (T, H, W, 3) uint8
    states:   np.ndarray,   # (T, STATE_DIM) float32
    actions:  np.ndarray,   # (T, ACTION_DIM) float32
    task_name:   str,
    instruction: str,
    success:     bool,
    extra_attrs: Optional[dict[str, str]] = None,
) -> None:
    """
    Save one episode to an HDF5 file.

    File layout
    ───────────
      /obs_image   (T, H, W, 3)  uint8   — compressed RGB frames
      /state       (T, 7)        float32 — robot state at each step
      /action      (T, 7)        float32 — action applied at each step
      /done        (T,)          bool    — True only at the final step

    Metadata (HDF5 attributes on the root group)
    ─────────────────────────────────────────────
      task_name, instruction, success, episode_len,
      state_dim, action_dim, img_height, img_width

    Compression note:
      gzip(4) reduces an 84×84 RGB trajectory of 50 frames from ~850 KB to
      ~50–150 KB depending on scene complexity.  For 500 episodes that is the
      difference between 425 MB and ~50 MB on disk.
    """
    T = len(frames)
    done = np.zeros(T, dtype=bool)
    done[-1] = True   # episode ends at the last timestep

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(filepath, "w") as f:
        f.create_dataset("obs_image", data=frames,  compression="gzip", compression_opts=4)
        f.create_dataset("state",     data=states,  compression="gzip", compression_opts=4)
        f.create_dataset("action",    data=actions, compression="gzip", compression_opts=4)
        f.create_dataset("done",      data=done)

        # Attributes — fast to read without loading full arrays
        f.attrs["task_name"]   = task_name
        f.attrs["instruction"] = instruction
        f.attrs["success"]     = bool(success)
        f.attrs["episode_len"] = T
        f.attrs["state_dim"]   = STATE_DIM
        f.attrs["action_dim"]  = ACTION_DIM
        f.attrs["img_height"]  = frames.shape[1]
        f.attrs["img_width"]   = frames.shape[2]
        if extra_attrs:
            for k, v in extra_attrs.items():
                f.attrs[k] = v


def _simulate_object_positions(
    states: np.ndarray,
    task: str,
    object_start_xyz: np.ndarray,
    box_xyz: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Build per-timestep object positions for rendering.

    This mirrors the rollout attachment/release logic used later in post-training
    and evaluation so generated episode videos show visible object motion.
    """
    positions: list[np.ndarray] = []
    object_xyz = object_start_xyz.astype(np.float32).copy()
    object_attached = task == "place_object"

    for state in states:
        eef_xyz = state[:3]
        gripper = float(state[6])

        if not object_attached and task in ("pick_object", "pick_and_place_object"):
            if gripper < 0.2 and np.linalg.norm(eef_xyz - object_xyz) < 0.08:
                object_attached = True

        if object_attached:
            object_xyz = eef_xyz.copy()
            if task in ("place_object", "pick_and_place_object") and gripper > 0.85:
                object_attached = False
                if box_xyz is not None:
                    object_xyz[2] = float(max(0.02, box_xyz[2]))

        positions.append(object_xyz.copy())

    return np.stack(positions, axis=0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# EPISODE GENERATION WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

def generate_episode(
    task: str,
    rng:  np.random.Generator,
    max_steps: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, bool, dict[str, str]]:
    """
    Generate one full episode for `task`.

    Returns (frames, states, actions, instruction, success, extra_attrs).
    """
    def rand_xyz() -> np.ndarray:
        return np.array([
            rng.uniform(*WORKSPACE["x"]),
            rng.uniform(*WORKSPACE["y"]),
            rng.uniform(*WORKSPACE["z"]),
        ], dtype=np.float32)

    instruction = ""
    extra_attrs: dict[str, str] = {}
    start_xyz   = rand_xyz()

    if task == "reach_target":
        instruction = str(rng.choice(INSTRUCTIONS[task]))
        target_xyz = rand_xyz()
        states, actions, success = oracle_reach_target(start_xyz, target_xyz, max_steps)
        frames = np.stack([
            render_frame(s, task, target_xyz=target_xyz) for s in states
        ])

    elif task == "pick_object":
        object_type = str(rng.choice(OBJECT_TYPES))
        template = str(rng.choice(INSTRUCTIONS[task]))
        instruction = template.format(object_name=object_type)
        object_xyz = np.array([
            rng.uniform(*WORKSPACE["x"]),
            rng.uniform(*WORKSPACE["y"]),
            rng.uniform(0.02, 0.06),
        ], dtype=np.float32)
        states, actions, success = oracle_pick_object(start_xyz, object_xyz, max_steps)
        object_positions = _simulate_object_positions(states, task, object_xyz)
        frames = np.stack([
            render_frame(s, task, object_xyz=obj_xyz, object_type=object_type)
            for s, obj_xyz in zip(states, object_positions)
        ])
        extra_attrs = {"object_type": object_type}

    elif task == "place_object":
        object_type = str(rng.choice(OBJECT_TYPES))
        template = str(rng.choice(INSTRUCTIONS[task]))
        instruction = template.format(object_name=object_type)

        box_xyz = np.array([
            rng.uniform(*WORKSPACE["x"]),
            rng.uniform(*WORKSPACE["y"]),
            rng.uniform(0.02, 0.06),
        ], dtype=np.float32)

        # Start anywhere in workspace with a grasped object.
        start_xyz = np.array([
            rng.uniform(*WORKSPACE["x"]),
            rng.uniform(*WORKSPACE["y"]),
            rng.uniform(0.20, 0.35),
        ], dtype=np.float32)
        object_xyz = start_xyz.copy()

        states, actions, success = oracle_place_object(start_xyz, box_xyz, max_steps=max_steps)
        object_positions = _simulate_object_positions(states, task, object_xyz, box_xyz=box_xyz)
        frames = np.stack([
            render_frame(s, task, object_xyz=obj_xyz, object_type=object_type, box_xyz=box_xyz)
            for s, obj_xyz in zip(states, object_positions)
        ])
        extra_attrs = {"object_type": object_type}

    elif task == "pick_and_place_object":
        object_type = str(rng.choice(OBJECT_TYPES))
        template = str(rng.choice(INSTRUCTIONS[task]))
        instruction = template.format(object_name=object_type)

        # Keep object and box near table height for a realistic pick/drop task.
        object_xyz = np.array([
            rng.uniform(*WORKSPACE["x"]),
            rng.uniform(*WORKSPACE["y"]),
            rng.uniform(0.02, 0.06),
        ], dtype=np.float32)

        # Ensure box is not too close to ball, otherwise task becomes trivial.
        while True:
            box_xyz = np.array([
                rng.uniform(*WORKSPACE["x"]),
                rng.uniform(*WORKSPACE["y"]),
                rng.uniform(0.02, 0.06),
            ], dtype=np.float32)
            if np.linalg.norm(object_xyz[:2] - box_xyz[:2]) > 0.18:
                break

        states, actions, success = oracle_pick_and_place_object(
            start_xyz, object_xyz, box_xyz, max_steps=max_steps
        )
        object_positions = _simulate_object_positions(states, task, object_xyz, box_xyz=box_xyz)
        frames = np.stack([
            render_frame(
                s, task, object_xyz=obj_xyz, object_type=object_type, box_xyz=box_xyz
            ) for s, obj_xyz in zip(states, object_positions)
        ])
        extra_attrs = {"object_type": object_type}

    else:
        raise ValueError(f"Unknown task: {task!r}")

    return frames, states, actions, instruction, success, extra_attrs


# ─────────────────────────────────────────────────────────────────────────────
# INSPECT HELPER  (run with --inspect to understand a saved file)
# ─────────────────────────────────────────────────────────────────────────────

def inspect_episode(path: Path) -> None:
    """Print the schema and first-frame statistics of a saved episode."""
    with h5py.File(path, "r") as f:
        print(f"\n{'─'*60}")
        print(f"  File : {path}")
        print(f"{'─'*60}")
        print("  Attributes (metadata):")
        for k, v in f.attrs.items():
            print(f"    {k:16s}: {v}")
        print("\n  Datasets:")
        for name, ds in f.items():
            print(f"    {name:12s}: shape={ds.shape}  dtype={ds.dtype}")
        print("\n  First timestep — state vector:")
        state0 = f["state"][0]
        labels = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
        for label, val in zip(labels, state0):
            print(f"    {label:8s}: {val:.4f}")
        print(f"\n  Action range over full episode:")
        actions = f["action"][:]
        for i, label in enumerate(labels):
            print(f"    d_{label:6s}: min={actions[:,i].min():.4f}  max={actions[:,i].max():.4f}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic VLA training data (Baby Step 2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tasks",        nargs="+", default=TASKS, choices=TASKS,
                        help="Which tasks to generate data for")
    parser.add_argument("--num-episodes", type=int, default=500,
                        help="Episodes to generate per task")
    parser.add_argument("--start-index",  type=int, default=0,
                        help="Starting episode index (use to append without overwriting)")
    parser.add_argument("--max-steps",    type=int, default=50,
                        help="Max timesteps per episode")
    parser.add_argument("--output-dir",   type=str, default="data/raw/sim_demos",
                        help="Root directory for HDF5 episode files")
    parser.add_argument("--seed",         type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--inspect",      action="store_true",
                        help="After generation, print schema of the first episode")
    args = parser.parse_args()

    rng        = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)

    print(f"\n[generate_sim_data]")
    print(f"  Tasks         : {args.tasks}")
    print(f"  Episodes/task : {args.num_episodes}")
    print(f"  Max steps     : {args.max_steps}")
    print(f"  Output dir    : {output_dir.resolve()}")
    print(f"  Seed          : {args.seed}")
    print()

    results = {}

    for task in args.tasks:
        task_dir = output_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)
        n_success = 0

        for ep_offset in tqdm(range(args.num_episodes), desc=f"  {task}", unit="ep"):
            ep_idx = args.start_index + ep_offset
            frames, states, actions, instruction, success, extra_attrs = generate_episode(
                task, rng, max_steps=args.max_steps
            )
            if success:
                n_success += 1

            save_episode(
                filepath    = task_dir / f"episode_{ep_idx:04d}.h5",
                frames      = frames,
                states      = states,
                actions     = actions,
                task_name   = task,
                instruction = instruction,
                success     = success,
                extra_attrs = extra_attrs,
            )

        results[task] = (n_success, args.num_episodes)
        pct = 100.0 * n_success / args.num_episodes
        print(f"  {task}: {n_success}/{args.num_episodes} successful  ({pct:.1f}%)")

    print(f"\n[generate_sim_data] Done.  Files saved to: {output_dir.resolve()}")

    # ── Print the data schema so it is easy to reference ──────────────────
    print(f"""
Episode HDF5 schema
───────────────────
  obs_image  (T, {IMG_H}, {IMG_W}, 3)   uint8    — RGB camera frames
  state      (T, {STATE_DIM})              float32  — [x, y, z, roll, pitch, yaw, gripper]
  action     (T, {ACTION_DIM})              float32  — [dx, dy, dz, d_roll, d_pitch, d_yaw, d_gripper]
  done       (T,)                bool     — True at final timestep

Attributes per file: task_name, instruction, success, episode_len
""")

    if args.inspect:
        first_ep = output_dir / args.tasks[0] / "episode_0000.h5"
        if first_ep.exists():
            inspect_episode(first_ep)


if __name__ == "__main__":
    main()
