"""
src/posttrain/dagger.py  -  Baby Step 9: Stage C post-training (DAgger-lite)

Goal
----
Improve robustness after SFT by rolling out the current policy, querying the
oracle on visited states, and fine-tuning on those correction samples.

This is a lightweight DAgger variant for the synthetic dataset in this repo:
- collect correction samples from policy rollouts in a simple simulator
- keep only high-error or failed states
- fine-tune the Step 8 policy on those oracle corrections
"""

from __future__ import annotations

import argparse
import hashlib
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from data.generate_sim_data import (
    ACTION_DIM,
    INSTRUCTIONS,
    OBJECT_TYPES,
    STATE_DIM,
    TASKS,
    WORKSPACE,
    _clip_to_workspace,
    render_frame,
)
from src.train.sft import VLABehaviorCloningModel, compute_action_loss


_TOKEN_RE = re.compile(r"[a-z0-9_]+")

# Placement checks should tolerate visual-control and accumulation error.
PLACEMENT_XY_TOL = 0.15
PLACEMENT_Z_MARGIN = 0.20
RELEASE_OPEN_THRESHOLD = 0.75


@dataclass
class CorrectionSample:
    image: torch.Tensor
    token_ids: torch.Tensor
    state: torch.Tensor
    action: torch.Tensor


@dataclass
class RolloutScene:
    task: str
    instruction: str
    state: np.ndarray
    object_xyz: np.ndarray | None = None
    object_type: str | None = None
    target_xyz: np.ndarray | None = None
    box_xyz: np.ndarray | None = None
    released_near_box: bool = False
    object_attached: bool = False
    object_start_xyz: np.ndarray | None = None
    lifted_after_pick: bool = False  # one-way flag: True once EE has cleared lift height after grasping


class CorrectionDataset(Dataset):
    def __init__(self, samples: list[CorrectionSample]) -> None:
        if not samples:
            raise RuntimeError("No correction samples available for DAgger fine-tuning")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> CorrectionSample:
        return self.samples[idx]


def collate_batch(items: list[CorrectionSample]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    images = torch.stack([item.image for item in items], dim=0)
    token_ids = torch.stack([item.token_ids for item in items], dim=0)
    states = torch.stack([item.state for item in items], dim=0)
    actions = torch.stack([item.action for item in items], dim=0)
    return images, token_ids, states, actions


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: str | Path) -> dict[str, Any]:
    cfg = OmegaConf.load(config_path)
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def tokenize_instruction(text: str, vocab_size: int, max_seq_len: int) -> torch.Tensor:
    words = _TOKEN_RE.findall(text.lower())
    token_ids = np.zeros(max_seq_len, dtype=np.int64)

    if not words:
        token_ids[0] = 1
        return torch.from_numpy(token_ids)

    for idx, token in enumerate(words[:max_seq_len]):
        digest = hashlib.sha1(token.encode("utf-8")).digest()
        hashed = int.from_bytes(digest[:4], byteorder="big", signed=False)
        token_ids[idx] = 2 + (hashed % max(1, vocab_size - 2))

    return torch.from_numpy(token_ids)


def rand_xyz(rng: np.random.Generator) -> np.ndarray:
    return np.array([
        rng.uniform(*WORKSPACE["x"]),
        rng.uniform(*WORKSPACE["y"]),
        rng.uniform(*WORKSPACE["z"]),
    ], dtype=np.float32)


def _get_goal_xyz(scene: "RolloutScene") -> np.ndarray:
    """Return the current sub-goal xyz for the rollout scene (matches oracle phase logic)."""
    state = scene.state
    thresh = 0.04

    if scene.task == "reach_target":
        assert scene.target_xyz is not None
        return scene.target_xyz.astype(np.float32).copy()

    if scene.task == "pick_object":
        assert scene.object_xyz is not None
        above_object = np.array([scene.object_xyz[0], scene.object_xyz[1], scene.object_xyz[2] + 0.15], dtype=np.float32)
        at_object = scene.object_xyz.astype(np.float32)
        lift_target = np.array([scene.object_xyz[0], scene.object_xyz[1],
                                 (scene.object_start_xyz[2] if scene.object_start_xyz is not None else scene.object_xyz[2]) + 0.25],
                                dtype=np.float32)
        if not scene.object_attached:  # pre-grasp: approaching
            # Only return to above_object if still clearly above or at hover height.
            # Once below above_object[2]-thresh, we've committed to the descent.
            need_approach = (
                np.linalg.norm(state[:3] - above_object) > thresh
                and state[2] >= above_object[2] - thresh
            )
            if need_approach:
                return above_object
            return at_object  # descending or closing gripper
        return lift_target  # grasped, lift up

    if scene.task == "place_object":
        assert scene.box_xyz is not None
        above_box = np.array([scene.box_xyz[0], scene.box_xyz[1], scene.box_xyz[2] + 0.18], dtype=np.float32)
        at_box_drop = np.array([scene.box_xyz[0], scene.box_xyz[1], scene.box_xyz[2] + 0.08], dtype=np.float32)
        retreat = np.array([scene.box_xyz[0], scene.box_xyz[1], scene.box_xyz[2] + 0.22], dtype=np.float32)
        # z-height guard: commit to descent once below above_box[2]-thresh.
        need_approach_box = (
            np.linalg.norm(state[:3] - above_box) > thresh
            and state[2] >= above_box[2] - thresh
        )
        if need_approach_box:
            return above_box
        elif np.linalg.norm(state[:3] - at_box_drop) > thresh:
            return at_box_drop
        elif not scene.object_attached:  # released — retreat upward
            return retreat
        return at_box_drop  # hovering while opening gripper

    if scene.task == "pick_and_place_object":
        assert scene.object_xyz is not None and scene.box_xyz is not None
        above_object = np.array([scene.object_xyz[0], scene.object_xyz[1], scene.object_xyz[2] + 0.15], dtype=np.float32)
        at_object = scene.object_xyz.astype(np.float32)
        orig_z = scene.object_start_xyz[2] if scene.object_start_xyz is not None else 0.04
        lift_after_pick = np.array([state[0], state[1], orig_z + 0.25], dtype=np.float32)
        above_box = np.array([scene.box_xyz[0], scene.box_xyz[1], scene.box_xyz[2] + 0.18], dtype=np.float32)
        at_box_drop = np.array([scene.box_xyz[0], scene.box_xyz[1], scene.box_xyz[2] + 0.08], dtype=np.float32)
        retreat = np.array([scene.box_xyz[0], scene.box_xyz[1], scene.box_xyz[2] + 0.22], dtype=np.float32)
        if not scene.object_attached:  # pre-grasp: approach object
            need_approach = (
                np.linalg.norm(state[:3] - above_object) > thresh
                and state[2] >= above_object[2] - thresh
            )
            if need_approach:
                return above_object
            return at_object
        else:
            # One-way lift gate: once we've cleared orig_z+0.20, we're committed to box approach.
            if not scene.lifted_after_pick:
                if state[2] < orig_z + 0.20:
                    return lift_after_pick
                scene.lifted_after_pick = True
            # Box approach: z-height guard prevents returning to above_box after descent.
            need_approach_box = (
                np.linalg.norm(state[:3] - above_box) > thresh
                and state[2] >= above_box[2] - thresh
            )
            if need_approach_box:
                return above_box
            elif np.linalg.norm(state[:3] - at_box_drop) > thresh:
                return at_box_drop
            else:
                return at_box_drop  # hover here while opening gripper / retreating

    return state[:3].copy()


def update_goal_delta(scene: "RolloutScene") -> None:
    """Recompute state[7:10] = goal_xyz - eef_xyz after each step."""
    goal = _get_goal_xyz(scene)
    scene.state[7:10] = goal - scene.state[:3]


def build_rollout_scene(task: str, rng: np.random.Generator) -> RolloutScene:
    instruction = ""
    state = np.zeros(STATE_DIM, dtype=np.float32)
    state[:3] = rand_xyz(rng)
    state[3:6] = 0.0
    state[6] = 1.0

    if task == "reach_target":
        instruction = str(rng.choice(INSTRUCTIONS[task]))
        target_xyz = rand_xyz(rng)
        scene = RolloutScene(task=task, instruction=instruction, state=state, target_xyz=target_xyz)
        update_goal_delta(scene)
        return scene

    if task == "pick_object":
        object_type = str(rng.choice(OBJECT_TYPES))
        instruction = str(rng.choice(INSTRUCTIONS[task])).format(object_name=object_type)
        object_xyz = np.array([
            rng.uniform(*WORKSPACE["x"]),
            rng.uniform(*WORKSPACE["y"]),
            rng.uniform(0.02, 0.06),
        ], dtype=np.float32)
        scene = RolloutScene(
            task=task,
            instruction=instruction,
            state=state,
            object_xyz=object_xyz,
            object_type=object_type,
            object_start_xyz=object_xyz.copy(),
        )
        update_goal_delta(scene)
        return scene

    if task == "place_object":
        object_type = str(rng.choice(OBJECT_TYPES))
        instruction = str(rng.choice(INSTRUCTIONS[task])).format(object_name=object_type)
        box_xyz = np.array([
            rng.uniform(*WORKSPACE["x"]),
            rng.uniform(*WORKSPACE["y"]),
            rng.uniform(0.02, 0.06),
        ], dtype=np.float32)
        state[:3] = np.array([
            rng.uniform(*WORKSPACE["x"]),
            rng.uniform(*WORKSPACE["y"]),
            rng.uniform(0.20, 0.35),
        ], dtype=np.float32)
        state[6] = 0.0
        object_xyz = state[:3].copy()
        scene = RolloutScene(
            task=task,
            instruction=instruction,
            state=state,
            object_xyz=object_xyz,
            object_type=object_type,
            box_xyz=box_xyz,
            object_attached=True,
            object_start_xyz=object_xyz.copy(),
        )
        update_goal_delta(scene)
        return scene

    if task == "pick_and_place_object":
        object_type = str(rng.choice(OBJECT_TYPES))
        instruction = str(rng.choice(INSTRUCTIONS[task])).format(object_name=object_type)
        object_xyz = np.array([
            rng.uniform(*WORKSPACE["x"]),
            rng.uniform(*WORKSPACE["y"]),
            rng.uniform(0.02, 0.06),
        ], dtype=np.float32)
        while True:
            box_xyz = np.array([
                rng.uniform(*WORKSPACE["x"]),
                rng.uniform(*WORKSPACE["y"]),
                rng.uniform(0.02, 0.06),
            ], dtype=np.float32)
            if np.linalg.norm(object_xyz[:2] - box_xyz[:2]) > 0.18:
                break
        scene = RolloutScene(
            task=task,
            instruction=instruction,
            state=state,
            object_xyz=object_xyz,
            object_type=object_type,
            box_xyz=box_xyz,
            object_start_xyz=object_xyz.copy(),
        )
        update_goal_delta(scene)
        return scene

    raise ValueError(f"Unknown task: {task}")


def rollout_frame(scene: RolloutScene) -> np.ndarray:
    return render_frame(
        scene.state,
        scene.task,
        target_xyz=scene.target_xyz,
        object_xyz=scene.object_xyz,
        object_type=scene.object_type,
        box_xyz=scene.box_xyz,
    )


def oracle_action_for_state(scene: RolloutScene) -> np.ndarray:
    state = scene.state
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    gain = 0.30
    thresh = 0.04

    if scene.task == "reach_target":
        assert scene.target_xyz is not None
        delta = scene.target_xyz - state[:3]
        if np.linalg.norm(delta) >= 0.05:
            action[:3] = 0.25 * delta
        return action

    if scene.task == "pick_object":
        assert scene.object_xyz is not None
        above_object = np.array([scene.object_xyz[0], scene.object_xyz[1], scene.object_xyz[2] + 0.15], dtype=np.float32)
        at_object = scene.object_xyz.astype(np.float32)
        lift_target = np.array([scene.object_xyz[0], scene.object_xyz[1], scene.object_xyz[2] + 0.25], dtype=np.float32)

        if not scene.object_attached:
            need_approach = (
                np.linalg.norm(state[:3] - above_object) > thresh
                and state[2] >= above_object[2] - thresh
            )
            if need_approach:
                action[:3] = gain * (above_object - state[:3])
            elif np.linalg.norm(state[:3] - at_object) > thresh:
                action[:3] = gain * (at_object - state[:3])
            else:
                action[6] = -0.4
        elif np.linalg.norm(state[:3] - lift_target) > thresh:
            action[:3] = gain * (lift_target - state[:3])
        return action

    if scene.task == "place_object":
        assert scene.box_xyz is not None
        above_box = np.array([scene.box_xyz[0], scene.box_xyz[1], scene.box_xyz[2] + 0.18], dtype=np.float32)
        at_box_drop = np.array([scene.box_xyz[0], scene.box_xyz[1], scene.box_xyz[2] + 0.08], dtype=np.float32)
        retreat = np.array([scene.box_xyz[0], scene.box_xyz[1], scene.box_xyz[2] + 0.22], dtype=np.float32)
        # z-height guard: only go to above_box if EE is still above it.
        need_approach_box = (
            np.linalg.norm(state[:3] - above_box) > thresh
            and state[2] >= above_box[2] - thresh
        )
        if need_approach_box:
            action[:3] = gain * (above_box - state[:3])
        elif np.linalg.norm(state[:3] - at_box_drop) > thresh:
            action[:3] = gain * (at_box_drop - state[:3])
        elif scene.object_attached:  # at drop pos, still holding — open gripper
            action[6] = +0.4
        else:  # released
            action[:3] = gain * (retreat - state[:3])
        return action

    if scene.task == "pick_and_place_object":
        assert scene.object_xyz is not None and scene.box_xyz is not None
        above_object = np.array([scene.object_xyz[0], scene.object_xyz[1], scene.object_xyz[2] + 0.15], dtype=np.float32)
        at_object = scene.object_xyz.astype(np.float32)
        orig_z = scene.object_start_xyz[2] if scene.object_start_xyz is not None else 0.04
        lift_after_pick = np.array([state[0], state[1], orig_z + 0.25], dtype=np.float32)
        above_box = np.array([scene.box_xyz[0], scene.box_xyz[1], scene.box_xyz[2] + 0.18], dtype=np.float32)
        at_box_drop = np.array([scene.box_xyz[0], scene.box_xyz[1], scene.box_xyz[2] + 0.08], dtype=np.float32)
        retreat = np.array([scene.box_xyz[0], scene.box_xyz[1], scene.box_xyz[2] + 0.22], dtype=np.float32)

        if not scene.object_attached:  # pre-grasp
            need_approach = (
                np.linalg.norm(state[:3] - above_object) > thresh
                and state[2] >= above_object[2] - thresh
            )
            if need_approach:
                action[:3] = gain * (above_object - state[:3])
            elif np.linalg.norm(state[:3] - at_object) > thresh:
                action[:3] = gain * (at_object - state[:3])
            else:
                action[6] = -0.4  # close gripper to grasp
        else:
            # One-way lift gate: once cleared orig_z+0.20, commit to box approach.
            if not scene.lifted_after_pick:
                if state[2] < orig_z + 0.20:
                    action[:3] = gain * (lift_after_pick - state[:3])
                    return action
                scene.lifted_after_pick = True
            need_approach_box = (
                np.linalg.norm(state[:3] - above_box) > thresh
                and state[2] >= above_box[2] - thresh
            )
            if need_approach_box:
                action[:3] = gain * (above_box - state[:3])
            elif np.linalg.norm(state[:3] - at_box_drop) > thresh:
                action[:3] = gain * (at_box_drop - state[:3])
            elif scene.object_attached:  # at drop pos, still holding — open
                action[6] = +0.4
            else:  # released
                action[:3] = gain * (retreat - state[:3])
        return action

    raise ValueError(f"Unknown task: {scene.task}")


def apply_policy_action(
    scene: RolloutScene,
    action: np.ndarray,
    action_clip: float,
    action_clip_gripper: float | None = None,
) -> None:
    clipped = np.asarray(action, dtype=np.float32).copy()
    clipped[:6] = np.clip(clipped[:6], -action_clip, action_clip)
    gripper_clip = action_clip if action_clip_gripper is None else action_clip_gripper
    clipped[6] = float(np.clip(clipped[6], -gripper_clip, gripper_clip))
    state = scene.state.copy()
    state[:3] += clipped[:3]
    state[3:6] += clipped[3:6]
    state[6] = float(np.clip(state[6] + clipped[6], 0.0, 1.0))
    state[:3] = _clip_to_workspace(state[:3])

    if scene.object_xyz is not None:
        # Grasp event: close enough to object and gripper mostly closed.
        if not scene.object_attached and state[6] < 0.2:
            grasp_dist = float(np.linalg.norm(state[:3] - scene.object_xyz[:3]))
            if grasp_dist < 0.08:
                scene.object_attached = True

        # One-way lift gate: set once EE has cleared the lift height after pick.
        if scene.object_attached and not scene.lifted_after_pick and scene.object_start_xyz is not None:
            orig_z = scene.object_start_xyz[2]
            if state[2] >= orig_z + 0.20:
                scene.lifted_after_pick = True

        # Attached objects move with the gripper.
        if scene.object_attached:
            scene.object_xyz = state[:3].copy()

            # Release event: gripper opens enough, object stays at current location.
            if state[6] > RELEASE_OPEN_THRESHOLD:
                scene.object_attached = False
                if scene.box_xyz is not None and np.linalg.norm(scene.object_xyz[:2] - scene.box_xyz[:2]) < PLACEMENT_XY_TOL:
                    scene.released_near_box = True
        else:
            # Detached object remains where it was released; mark successful placement near box.
            if scene.box_xyz is not None and np.linalg.norm(scene.object_xyz[:2] - scene.box_xyz[:2]) < PLACEMENT_XY_TOL:
                if state[6] > RELEASE_OPEN_THRESHOLD:
                    scene.released_near_box = True

    scene.state = state


def is_success(scene: RolloutScene) -> bool:
    state = scene.state

    if scene.task == "reach_target":
        assert scene.target_xyz is not None
        return bool(np.linalg.norm(scene.target_xyz - state[:3]) < 0.05)

    if scene.task == "pick_object":
        assert scene.object_xyz is not None
        start_z = scene.object_start_xyz[2] if scene.object_start_xyz is not None else scene.object_xyz[2]
        return bool(scene.object_xyz[2] > start_z + 0.08)

    if scene.task == "place_object":
        assert scene.box_xyz is not None
        assert scene.object_xyz is not None
        return bool(
            scene.released_near_box
            and not scene.object_attached
            and scene.object_xyz[2] <= scene.box_xyz[2] + PLACEMENT_Z_MARGIN
        )

    if scene.task == "pick_and_place_object":
        assert scene.box_xyz is not None
        assert scene.object_xyz is not None
        return bool(
            scene.released_near_box
            and not scene.object_attached
            and scene.object_xyz[2] <= scene.box_xyz[2] + PLACEMENT_Z_MARGIN
        )

    return False


def collect_corrections(
    model: VLABehaviorCloningModel,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[list[CorrectionSample], dict[str, float]]:
    model.eval()
    rng = np.random.default_rng(int(config["experiment"]["seed"]))

    tasks = config["collect"].get("tasks", TASKS)
    num_episodes = int(config["collect"]["num_rollout_episodes"])
    max_steps = int(config["collect"]["max_steps"])
    correction_mae_threshold = float(config["collect"]["correction_mae_threshold"])
    action_clip = float(config["collect"]["action_clip"])
    action_clip_gripper = float(config["collect"].get("action_clip_gripper", action_clip))
    keep_success_all_steps = bool(config["collect"].get("keep_success_all_steps", True))
    vocab_size = int(config["model"]["vocab_size"])
    max_seq_len = int(config["model"]["max_seq_len"])

    samples: list[CorrectionSample] = []
    successes = 0
    total_mae = 0.0
    total_steps = 0

    for _ in tqdm(range(num_episodes), desc="Collect corrections", leave=False):
        task = str(rng.choice(tasks))
        scene = build_rollout_scene(task, rng)
        high_error_samples: list[CorrectionSample] = []
        all_step_samples: list[CorrectionSample] = []
        episode_succeeded = False

        for _step in range(max_steps):
            frame = rollout_frame(scene)
            image = torch.from_numpy(frame).permute(2, 0, 1).float().div(255.0)
            token_ids = tokenize_instruction(scene.instruction, vocab_size, max_seq_len)

            with torch.no_grad():
                pred_action = model(
                    image.unsqueeze(0).to(device),
                    token_ids.unsqueeze(0).to(device),
                    torch.from_numpy(scene.state.copy()).unsqueeze(0).to(device),
                )[0].detach().cpu()

            oracle_action = torch.from_numpy(oracle_action_for_state(scene))
            step_mae = float(torch.mean(torch.abs(pred_action - oracle_action)).item())
            total_mae += step_mae
            total_steps += 1

            sample = CorrectionSample(
                image=image,
                token_ids=token_ids,
                state=torch.from_numpy(scene.state.copy()),
                action=oracle_action.float(),
            )
            all_step_samples.append(sample)

            if step_mae >= correction_mae_threshold:
                high_error_samples.append(sample)

            apply_policy_action(
                scene,
                pred_action.numpy(),
                action_clip,
                action_clip_gripper=action_clip_gripper,
            )
            update_goal_delta(scene)

            if is_success(scene):
                successes += 1
                episode_succeeded = True
                break

        if episode_succeeded:
            if keep_success_all_steps:
                samples.extend(all_step_samples)
            else:
                samples.extend(high_error_samples)
        else:
            samples.extend(high_error_samples)

    metrics = {
        "num_samples": float(len(samples)),
        "success_rate": float(successes / max(1, num_episodes)),
        "mean_rollout_mae": float(total_mae / max(1, total_steps)),
    }
    return samples, metrics


def save_corrections(path: Path, samples: list[CorrectionSample], metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "images": torch.stack([sample.image for sample in samples], dim=0) if samples else torch.empty(0, 3, 84, 84),
        "token_ids": torch.stack([sample.token_ids for sample in samples], dim=0) if samples else torch.empty(0, dtype=torch.long),
        "states": torch.stack([sample.state for sample in samples], dim=0) if samples else torch.empty(0, STATE_DIM),
        "actions": torch.stack([sample.action for sample in samples], dim=0) if samples else torch.empty(0, ACTION_DIM),
        "metrics": metrics,
    }
    torch.save(payload, path)


def evaluate(
    model: VLABehaviorCloningModel,
    loader: DataLoader,
    device: torch.device,
    loss_name: str,
    huber_delta: float,
) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    maes: list[float] = []
    with torch.no_grad():
        for images, token_ids, states, actions in loader:
            images = images.to(device, non_blocking=True)
            token_ids = token_ids.to(device, non_blocking=True)
            states = states.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            preds = model(images, token_ids, states)
            loss = compute_action_loss(preds, actions, loss_name, huber_delta)
            mae = torch.mean(torch.abs(preds - actions))
            losses.append(float(loss.item()))
            maes.append(float(mae.item()))
    return float(np.mean(losses)), float(np.mean(maes))


def train(config: dict[str, Any]) -> None:
    seed = int(config["experiment"]["seed"])
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(config["experiment"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    model = VLABehaviorCloningModel(
        model_size=str(config["model"]["size"]),
        num_fusion_layers=int(config["model"]["num_fusion_layers"]),
        dropout=float(config["model"]["dropout"]),
        action_dim=int(config["model"]["action_dim"]),
        use_state=bool(config["model"].get("use_state", False)),
        state_dim=int(config["model"].get("state_dim", 7)),
    ).to(device)

    sft_ckpt = Path(config["init"]["sft_checkpoint"])
    if not sft_ckpt.exists():
        raise FileNotFoundError(f"SFT checkpoint not found: {sft_ckpt}")

    checkpoint = torch.load(sft_ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    print(f"Loaded Step 8 policy : {sft_ckpt}")

    # Store frozen SFT weights for BC regularization during fine-tuning
    bc_reg_weight = float(config["train"].get("bc_reg_weight", 0.0))
    sft_params: list[torch.Tensor] | None = None
    if bc_reg_weight > 0.0:
        sft_params = [p.detach().clone() for p in model.parameters()]
        print(f"BC regularization    : L2 weight={bc_reg_weight:.2e}")

    samples, rollout_metrics = collect_corrections(model, config, device)
    corrections_path = out_dir / "corrections.pt"
    save_corrections(corrections_path, samples, rollout_metrics)

    print("\n[Step 9] DAgger-lite collection")
    print("=" * 72)
    print(f"Collected corrections: {int(rollout_metrics['num_samples'])}")
    print(f"Rollout success rate : {rollout_metrics['success_rate']:.3f}")
    print(f"Rollout mean MAE     : {rollout_metrics['mean_rollout_mae']:.4f}")
    print(f"Saved corrections    : {corrections_path}")

    if not samples:
        print("No correction samples collected; keeping Step 8 checkpoint as-is.")
        return

    dataset = CorrectionDataset(samples)
    val_ratio = float(config["train"]["val_ratio"])
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        train_size = len(dataset) - 1
        val_size = 1
    train_set, val_set = random_split(dataset, [train_size, val_size])

    batch_size = int(config["train"]["batch_size"])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=False, collate_fn=collate_batch)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, drop_last=False, collate_fn=collate_batch)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optim"]["lr"]),
        betas=(float(config["optim"]["beta1"]), float(config["optim"]["beta2"])),
        weight_decay=float(config["optim"]["weight_decay"]),
    )

    scheduler_name = str(config["optim"].get("scheduler", "none")).lower()
    scheduler: torch.optim.lr_scheduler.LRScheduler | None
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(config["train"]["epochs"])),
            eta_min=float(config["optim"].get("min_lr", 1e-5)),
        )
    elif scheduler_name == "none":
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    epochs = int(config["train"]["epochs"])
    grad_clip_norm = float(config["train"].get("grad_clip_norm", 1.0))
    loss_name = str(config["train"].get("loss_name", "huber")).lower()
    huber_delta = float(config["train"].get("huber_delta", 0.25))
    early_stopping_patience = int(config["train"].get("early_stopping_patience", 0))
    early_stopping_min_delta = float(config["train"].get("early_stopping_min_delta", 0.0))

    print("\n[Step 9] DAgger-lite fine-tuning")
    print("=" * 72)
    print(f"Correction split      : train={len(train_set)} val={len(val_set)}")
    print(f"Batch size            : {batch_size}")
    print(f"Loss                  : {loss_name}")
    print(f"Scheduler             : {scheduler_name}")

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        train_maes: list[float] = []
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)

        for images, token_ids, states, actions in progress:
            images = images.to(device, non_blocking=True)
            token_ids = token_ids.to(device, non_blocking=True)
            states = states.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            preds = model(images, token_ids, states)
            loss = compute_action_loss(preds, actions, loss_name, huber_delta)
            if sft_params is not None:
                l2_reg = sum(
                    (p - p_ref).pow(2).sum()
                    for p, p_ref in zip(model.parameters(), sft_params)
                )
                loss = loss + bc_reg_weight * l2_reg
            mae = torch.mean(torch.abs(preds - actions))
            loss.backward()

            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            optimizer.step()
            train_losses.append(float(loss.item()))
            train_maes.append(float(mae.item()))
            progress.set_postfix(loss=f"{train_losses[-1]:.4f}", mae=f"{train_maes[-1]:.4f}")

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        train_mae = float(np.mean(train_maes)) if train_maes else float("nan")
        val_loss, val_mae = evaluate(model, val_loader, device, loss_name, huber_delta)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_mae={train_mae:.4f} | "
            f"val_loss={val_loss:.4f} val_mae={val_mae:.4f}"
        )

        if scheduler is not None:
            scheduler.step()
            print(f"  LR after scheduler step: {optimizer.param_groups[0]['lr']:.8f}")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "best_val_loss": best_val_loss,
            "rollout_metrics": rollout_metrics,
        }, out_dir / "last.pt")

        if val_loss < (best_val_loss - early_stopping_min_delta):
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "best_val_loss": best_val_loss,
                "rollout_metrics": rollout_metrics,
            }, out_dir / "best.pt")
            print(f"  Saved new best checkpoint: {out_dir / 'best.pt'}")
        else:
            epochs_without_improvement += 1

        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            print(
                f"  Early stopping triggered after {epochs_without_improvement} "
                f"epoch(s) without improvement."
            )
            break

    print("\n[Step 9] Post-training complete.")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints : {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage C DAgger-lite post-training")
    parser.add_argument("--config", type=str, default="configs/posttrain.yaml", help="Path to post-training YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()