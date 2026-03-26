"""
src/train/sft.py  -  Baby Step 8: Stage B Supervised Fine-Tuning (behavior cloning)

Goal
----
Train the full VLA policy to imitate expert actions from the synthetic HDF5 demos.

Given a timestep sample (image_t, instruction, action_t), predict the expert
7D action delta with the full model:
  VisionEncoder -> LanguageEncoder -> Fusion -> ActionHead

Stage A pre-training can be used to warm-start the vision and language encoders.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from src.models.action_head import ACTION_HEAD_CONFIGS, build_action_head
from src.models.fusion import build_fusion_module
from src.models.language_encoder import build_language_encoder
from src.models.vision_encoder import build_vision_encoder


_TOKEN_RE = re.compile(r"[a-z0-9_]+")


@dataclass
class Batch:
    images: torch.Tensor
    token_ids: torch.Tensor
    states: torch.Tensor
    actions: torch.Tensor


class H5TimestepBehaviorCloningDataset(Dataset):
    """
    Flattens episode files into timestep-level training samples.

    Each item is one expert timestep:
      image_t + instruction -> action_t
    """

    def __init__(
        self,
        episode_files: list[Path],
        vocab_size: int,
        max_seq_len: int,
        use_state: bool,
        state_dim: int,
        min_action_l1: float,
        keep_low_action_prob: float,
        seed: int,
    ) -> None:
        if not episode_files:
            raise RuntimeError("No episode files provided to SFT dataset")

        self.episode_files = episode_files
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.use_state = use_state
        self.state_dim = state_dim
        self.min_action_l1 = min_action_l1
        self.keep_low_action_prob = keep_low_action_prob
        self.rng = random.Random(seed)
        self.index: list[tuple[Path, int]] = []

        for path in self.episode_files:
            with h5py.File(path, "r") as f:
                actions = np.asarray(f["action"][:], dtype=np.float32)

            for timestep, action in enumerate(actions):
                action_l1 = float(np.mean(np.abs(action)))
                keep = action_l1 >= self.min_action_l1 or self.rng.random() < self.keep_low_action_prob
                if keep:
                    self.index.append((path, timestep))

        if not self.index:
            raise RuntimeError("All SFT timesteps were filtered out; relax action filtering in config")

    def __len__(self) -> int:
        return len(self.index)

    def _tokenize(self, text: str) -> torch.Tensor:
        words = _TOKEN_RE.findall(text.lower())
        token_ids = np.zeros(self.max_seq_len, dtype=np.int64)

        if not words:
            token_ids[0] = 1
            return torch.from_numpy(token_ids)

        for idx, token in enumerate(words[: self.max_seq_len]):
            digest = hashlib.sha1(token.encode("utf-8")).digest()
            hashed = int.from_bytes(digest[:4], byteorder="big", signed=False)
            token_ids[idx] = 2 + (hashed % max(1, self.vocab_size - 2))

        return torch.from_numpy(token_ids)

    def __getitem__(self, idx: int) -> Batch:
        path, timestep = self.index[idx]

        with h5py.File(path, "r") as f:
            image = f["obs_image"][timestep]
            action = f["action"][timestep]
            if "state" in f:
                state = f["state"][timestep]
            else:
                state = np.zeros(self.state_dim, dtype=np.float32)
            instruction = str(f.attrs.get("instruction", ""))

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)
        token_ids = self._tokenize(instruction)
        state_tensor = torch.from_numpy(np.asarray(state, dtype=np.float32))
        if not self.use_state:
            state_tensor = torch.zeros_like(state_tensor)
        action_tensor = torch.from_numpy(np.asarray(action, dtype=np.float32))

        return Batch(
            images=image_tensor,
            token_ids=token_ids,
            states=state_tensor,
            actions=action_tensor,
        )


def collate_batch(items: list[Batch]) -> Batch:
    return Batch(
        images=torch.stack([item.images for item in items], dim=0),
        token_ids=torch.stack([item.token_ids for item in items], dim=0),
        states=torch.stack([item.states for item in items], dim=0),
        actions=torch.stack([item.actions for item in items], dim=0),
    )


class VLABehaviorCloningModel(nn.Module):
    def __init__(
        self,
        model_size: str,
        num_fusion_layers: int,
        dropout: float,
        action_dim: int,
        use_state: bool,
        state_dim: int,
    ) -> None:
        super().__init__()
        self.vision = build_vision_encoder(model_size, dropout=dropout)
        self.language = build_language_encoder(model_size, dropout=dropout)
        self.fusion = build_fusion_module(model_size, num_fusion_layers=num_fusion_layers, dropout=dropout)
        self.action_head = build_action_head(model_size, action_dim=action_dim, dropout=dropout)
        self.use_state = use_state

        if self.use_state:
            fused_dim = int(ACTION_HEAD_CONFIGS[model_size]["fused_dim"])
            self.state_proj = nn.Sequential(
                nn.LayerNorm(state_dim),
                nn.Linear(state_dim, fused_dim),
            )
        else:
            self.state_proj = None

    def forward(self, images: torch.Tensor, token_ids: torch.Tensor, states: torch.Tensor | None = None) -> torch.Tensor:
        vision_cls, vision_tokens = self.vision(images)
        language_cls, language_tokens = self.language(token_ids)
        fused_cls, _ = self.fusion(vision_cls, vision_tokens, language_cls, language_tokens)

        if self.use_state:
            if states is None:
                raise ValueError("states must be provided when use_state=True")
            fused_cls = fused_cls + self.state_proj(states)

        return self.action_head(fused_cls)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: str | Path) -> dict[str, Any]:
    cfg = OmegaConf.load(config_path)
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def split_episode_files(all_files: list[Path], val_ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
    rng = random.Random(seed)
    files = all_files.copy()
    rng.shuffle(files)
    val_size = max(1, int(len(files) * val_ratio))
    val_files = files[:val_size]
    train_files = files[val_size:]
    if not train_files:
        raise RuntimeError("Validation split consumed all episode files")
    return train_files, val_files


def move_batch(batch: Batch, device: torch.device) -> Batch:
    return Batch(
        images=batch.images.to(device, non_blocking=True),
        token_ids=batch.token_ids.to(device, non_blocking=True),
        states=batch.states.to(device, non_blocking=True),
        actions=batch.actions.to(device, non_blocking=True),
    )


def compute_action_loss(
    pred_actions: torch.Tensor,
    target_actions: torch.Tensor,
    loss_name: str,
    huber_delta: float,
) -> torch.Tensor:
    if loss_name == "mse":
        return F.mse_loss(pred_actions, target_actions)
    if loss_name == "huber":
        return F.huber_loss(pred_actions, target_actions, delta=huber_delta)
    raise ValueError(f"Unsupported loss_name: {loss_name}")


def evaluate(
    model: VLABehaviorCloningModel,
    loader: DataLoader,
    device: torch.device,
    loss_name: str,
    huber_delta: float,
    max_eval_batches: int,
) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    maes: list[float] = []

    with torch.no_grad():
        for step, batch in enumerate(loader):
            if max_eval_batches > 0 and step >= max_eval_batches:
                break

            batch = move_batch(batch, device)
            pred_actions = model(batch.images, batch.token_ids, batch.states)
            loss = compute_action_loss(pred_actions, batch.actions, loss_name, huber_delta)
            mae = torch.mean(torch.abs(pred_actions - batch.actions))
            losses.append(float(loss.item()))
            maes.append(float(mae.item()))

    if not losses:
        return float("nan"), float("nan")

    return float(np.mean(losses)), float(np.mean(maes))


def save_checkpoint(
    out_dir: Path,
    name: str,
    model: VLABehaviorCloningModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
    best_val_loss: float,
) -> None:
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "best_val_loss": best_val_loss,
    }
    torch.save(ckpt, out_dir / name)


def maybe_load_pretrained_backbone(
    model: VLABehaviorCloningModel,
    checkpoint_path: str | Path | None,
    device: torch.device,
) -> None:
    if checkpoint_path is None:
        print("Pretrained init      : disabled")
        return

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Pretrain checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    vision_state = {
        key.replace("vision.", "", 1): value
        for key, value in state_dict.items()
        if key.startswith("vision.")
    }
    language_state = {
        key.replace("language.", "", 1): value
        for key, value in state_dict.items()
        if key.startswith("language.")
    }

    current_vision = model.vision.state_dict()
    current_language = model.language.state_dict()

    compatible_vision_state = {
        key: value
        for key, value in vision_state.items()
        if key in current_vision and current_vision[key].shape == value.shape
    }
    compatible_language_state = {
        key: value
        for key, value in language_state.items()
        if key in current_language and current_language[key].shape == value.shape
    }

    vision_msg = model.vision.load_state_dict(compatible_vision_state, strict=False)
    language_msg = model.language.load_state_dict(compatible_language_state, strict=False)

    print(f"Pretrained init      : loaded from {ckpt_path}")
    print(f"  Vision loaded keys  : {len(compatible_vision_state)} / {len(current_vision)}")
    print(f"  Vision missing keys : {len(vision_msg.missing_keys)}")
    print(f"  Vision unexpected   : {len(vision_msg.unexpected_keys)}")
    print(f"  Language loaded keys: {len(compatible_language_state)} / {len(current_language)}")
    print(f"  Language missing    : {len(language_msg.missing_keys)}")
    print(f"  Language unexpected : {len(language_msg.unexpected_keys)}")


def train(config: dict[str, Any]) -> None:
    seed = int(config["experiment"]["seed"])
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(config["experiment"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    root_dir = Path(config["data"]["root_dir"])
    if not root_dir.exists():
        raise FileNotFoundError(f"Dataset root not found: {root_dir}")

    all_files = sorted(root_dir.rglob("*.h5"))
    max_files = int(config["data"].get("max_files", 0))
    if max_files > 0:
        all_files = all_files[:max_files]
    if not all_files:
        raise RuntimeError(f"No .h5 files found under: {root_dir}")

    train_files, val_files = split_episode_files(
        all_files=all_files,
        val_ratio=float(config["data"]["val_ratio"]),
        seed=seed,
    )

    train_set = H5TimestepBehaviorCloningDataset(
        episode_files=train_files,
        vocab_size=int(config["model"]["vocab_size"]),
        max_seq_len=int(config["model"]["max_seq_len"]),
        use_state=bool(config["model"].get("use_state", False)),
        state_dim=int(config["model"].get("state_dim", 7)),
        min_action_l1=float(config["data"].get("min_action_l1", 0.0)),
        keep_low_action_prob=float(config["data"].get("keep_low_action_prob", 1.0)),
        seed=seed,
    )
    val_set = H5TimestepBehaviorCloningDataset(
        episode_files=val_files,
        vocab_size=int(config["model"]["vocab_size"]),
        max_seq_len=int(config["model"]["max_seq_len"]),
        use_state=bool(config["model"].get("use_state", False)),
        state_dim=int(config["model"].get("state_dim", 7)),
        min_action_l1=float(config["data"].get("min_action_l1", 0.0)),
        keep_low_action_prob=float(config["data"].get("keep_low_action_prob", 1.0)),
        seed=seed + 1,
    )

    batch_size = int(config["data"]["batch_size"])

    # Task-balanced sampling: equalise gradient updates across tasks regardless
    # of how many steps each task contributes (p&p has ~5x more steps than pick).
    if config["data"].get("task_balanced_sampling", False):
        task_counts: dict[str, int] = {}
        sample_tasks: list[str] = []
        for file_path, _ in train_set.index:
            task = Path(file_path).parent.name
            sample_tasks.append(task)
            task_counts[task] = task_counts.get(task, 0) + 1
        sample_weights = [1.0 / task_counts[t] for t in sample_tasks]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        print(f"Task-balanced sampling: {task_counts}")
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=int(config["data"]["num_workers"]),
            pin_memory=bool(config["data"]["pin_memory"]),
            drop_last=True,
            collate_fn=collate_batch,
        )
    else:
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=int(config["data"]["num_workers"]),
            pin_memory=bool(config["data"]["pin_memory"]),
            drop_last=True,
            collate_fn=collate_batch,
        )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=bool(config["data"]["pin_memory"]),
        drop_last=False,
        collate_fn=collate_batch,
    )

    model = VLABehaviorCloningModel(
        model_size=str(config["model"]["size"]),
        num_fusion_layers=int(config["model"]["num_fusion_layers"]),
        dropout=float(config["model"]["dropout"]),
        action_dim=int(config["model"]["action_dim"]),
        use_state=bool(config["model"].get("use_state", False)),
        state_dim=int(config["model"].get("state_dim", 7)),
    ).to(device)

    maybe_load_pretrained_backbone(
        model=model,
        checkpoint_path=config["init"].get("pretrain_checkpoint"),
        device=device,
    )

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
    max_train_batches = int(config["train"].get("max_train_batches", 0))
    max_eval_batches = int(config["train"].get("max_eval_batches", 0))
    grad_clip_norm = float(config["train"].get("grad_clip_norm", 1.0))
    loss_name = str(config["train"].get("loss_name", "huber")).lower()
    huber_delta = float(config["train"].get("huber_delta", 1.0))
    early_stopping_patience = int(config["train"].get("early_stopping_patience", 0))
    early_stopping_min_delta = float(config["train"].get("early_stopping_min_delta", 0.0))

    print("\n[Step 8] Stage B supervised fine-tuning")
    print("=" * 72)
    print(f"Device               : {device}")
    print(f"Dataset root         : {root_dir}")
    print(f"Episode files        : train={len(train_files)} val={len(val_files)}")
    print(f"Timestep samples     : train={len(train_set)} val={len(val_set)}")
    print(f"Batch size           : {batch_size}")
    print(f"Action filter        : min_l1={float(config['data'].get('min_action_l1', 0.0))}, keep_low={float(config['data'].get('keep_low_action_prob', 1.0))}")
    print(f"Model size           : {config['model']['size']}")
    print(f"Fusion layers        : {config['model']['num_fusion_layers']}")
    print(f"Use state input      : {bool(config['model'].get('use_state', False))}")
    print(f"Loss                 : {loss_name}")
    print(f"Scheduler            : {scheduler_name}")
    print(f"Early stopping       : patience={early_stopping_patience}, min_delta={early_stopping_min_delta}")
    print(f"Trainable parameters : {model.num_parameters():,}")

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        train_maes: list[float] = []

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for step, batch in enumerate(progress):
            if max_train_batches > 0 and step >= max_train_batches:
                break

            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)

            pred_actions = model(batch.images, batch.token_ids, batch.states)
            loss = compute_action_loss(pred_actions, batch.actions, loss_name, huber_delta)
            mae = torch.mean(torch.abs(pred_actions - batch.actions))
            loss.backward()

            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            optimizer.step()

            train_losses.append(float(loss.item()))
            train_maes.append(float(mae.item()))
            progress.set_postfix(loss=f"{train_losses[-1]:.4f}", mae=f"{train_maes[-1]:.4f}")

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        train_mae = float(np.mean(train_maes)) if train_maes else float("nan")

        val_loss, val_mae = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            loss_name=loss_name,
            huber_delta=huber_delta,
            max_eval_batches=max_eval_batches,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_mae={train_mae:.4f} | "
            f"val_loss={val_loss:.4f} val_mae={val_mae:.4f}"
        )

        if scheduler is not None:
            scheduler.step()
            print(f"  LR after scheduler step: {optimizer.param_groups[0]['lr']:.8f}")

        save_checkpoint(
            out_dir=out_dir,
            name="last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            config=config,
            best_val_loss=best_val_loss,
        )

        if val_loss < (best_val_loss - early_stopping_min_delta):
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_checkpoint(
                out_dir=out_dir,
                name="best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=config,
                best_val_loss=best_val_loss,
            )
            print(f"  Saved new best checkpoint: {out_dir / 'best.pt'}")
        else:
            epochs_without_improvement += 1

        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            print(
                f"  Early stopping triggered after {epochs_without_improvement} "
                f"epoch(s) without improvement."
            )
            break

    print("\n[Step 8] SFT complete.")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints : {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage B supervised fine-tuning")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sft.yaml",
        help="Path to SFT YAML config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
