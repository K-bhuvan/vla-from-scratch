"""
src/models/action_head.py  —  Baby Step 6: Action Head (Policy MLP from scratch)

════════════════════════════════════════════════════════════════════════════════
WHAT THIS FILE BUILDS
════════════════════════════════════════════════════════════════════════════════
A policy network (MLP) that maps the fused vision-language representation to
robot action deltas.

    Input  : (B, fused_dim)  — fused_cls from VisionLanguageFusion
    Output : (B, ACTION_DIM) — 7 continuous action deltas

The 7 output dimensions [dx, dy, dz, d_roll, d_pitch, d_yaw, d_gripper]:
  dx, dy, dz       — end-effector Cartesian translation (metres per step)
  d_roll, d_pitch, d_yaw — end-effector orientation change (radians per step)
  d_gripper        — gripper openness change (−1 = close fully, +1 = open fully)

This is the final module in the forward pass.  Every perception module output
flows into this one head that decides "what should the robot do next?"

════════════════════════════════════════════════════════════════════════════════
WHY AN MLP (NOT ANOTHER TRANSFORMER)?
════════════════════════════════════════════════════════════════════════════════
The fused_cls vector is already a rich, contextualised representation.  At this
point the task is pure regression: map a vector to 7 numbers.  MLPs are the
right tool for this — simple, fast, and expressive enough for the job.

Transformers are powerful at learning relationships between tokens in sequences.
Here we have a single vector as input (no sequence), so a Transformer adds
overhead without benefit.  All RT-2, OpenVLA, and similar models use an MLP
for the final action regression head.

════════════════════════════════════════════════════════════════════════════════
ARCHITECTURE OPTIONS
════════════════════════════════════════════════════════════════════════════════
  ActionHead (default): 2-layer MLP
    fused_dim → hidden_dim → ACTION_DIM
    Uses LayerNorm + GELU for stability.

  DeepActionHead (optional): 3-layer MLP
    fused_dim → hidden_dim → hidden_dim//2 → ACTION_DIM
    More capacity for complex tasks; marginal improvement for MVP.

════════════════════════════════════════════════════════════════════════════════
DATA FLOW DIAGRAM
════════════════════════════════════════════════════════════════════════════════

  fused_cls : (B, 2 * embed_dim)   — from VisionLanguageFusion
        │
        ▼
  LayerNorm(fused_dim)              — stabilise input activations
        │
        ▼
  Linear(fused_dim → hidden_dim)
        │
        ▼
  GELU()                            — smooth non-linearity
        │
        ▼
  Dropout(p)                        — regularisation
        │
        ▼
  Linear(hidden_dim → ACTION_DIM)   — project to 7 action deltas
        │
        ▼
  action : (B, 7)
    [dx, dy, dz, d_roll, d_pitch, d_yaw, d_gripper]

════════════════════════════════════════════════════════════════════════════════
FULL FORWARD PASS: all 4 modules together
════════════════════════════════════════════════════════════════════════════════

  image (B, 3, 84, 84)          token_ids (B, seq_len)
        │                               │
        ▼                               ▼
  VisionEncoder                 LanguageEncoder
  (Step 3)                      (Step 4)
        │                               │
  v_cls (B, E)                   l_cls (B, E)
  v_tokens (B, 144, E)           l_tokens (B, S, E)
        │                               │
        └───────────────────────────────┘
                        │
                        ▼
              VisionLanguageFusion
              (Step 5)
                        │
               fused_cls (B, 2E)
               fused_tokens (B, 144, 2E)
                        │
                        ▼
                  ActionHead
                  (Step 6)    ←── YOU ARE HERE
                        │
                   action (B, 7)
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

ACTION_DIM = 7  # [dx, dy, dz, d_roll, d_pitch, d_yaw, d_gripper]

# Maps which config name → fused_dim (= 2 * embed_dim from the fusion module)
ACTION_HEAD_CONFIGS: dict[str, dict] = {
    # fused_dim = 2 * embed_dim (from fusion module concatenation)
    # hidden_dim = fused_dim * 2  (default: expand then project)
    "tiny":  dict(fused_dim=256,  hidden_dim=512),   # tiny vision+lang → 2*128=256
    "small": dict(fused_dim=512,  hidden_dim=1024),  # small vision+lang → 2*256=512
    "base":  dict(fused_dim=1024, hidden_dim=2048),  # base vision+lang → 2*512=1024
}


# ─────────────────────────────────────────────────────────────────────────────
# ACTION HEAD: 2-LAYER MLP POLICY
# ─────────────────────────────────────────────────────────────────────────────

class ActionHead(nn.Module):
    """
    2-layer MLP that regresses the fused multimodal representation to 7 action
    deltas.

    INTUITION
    ─────────
    At this point, fused_cls already "knows":
      - What the robot sees (from vision encoder)
      - What the instruction says (from language encoder)
      - How image regions relate to words (from fusion)
      - What the robot's state is (can be added as extra input, see below)

    The MLP's only job is to translate that rich understanding into numbers:
    "Given everything I know, how much should the arm move in each direction?"

    WHY LAYER NORM AT THE INPUT?
    ─────────────────────────────
    fused_cls is the concatenation of two vectors:
        vision_cls (embed_dim)  ||  language_cls (embed_dim)
    These two halves might live at very different numerical scales (one modality
    may be trained faster than the other early in training).  LayerNorm at the
    input normalises them to the same scale before the MLP sees them.

    WHY NO TANH OUTPUT ACTIVATION?
    ───────────────────────────────
    We don't clip the output with tanh because:
    1. During training, action targets are already normalised delta values in
       roughly [−1, 1].  The loss will naturally push outputs into that range.
    2. tanh saturates at ±1 and has near-zero gradients at the extremes, which
       can stall training.
    3. At inference, we clip actions to safe workspace bounds in the env wrapper
       (not inside the model).
    """

    def __init__(
        self,
        fused_dim:  int,
        hidden_dim: int,
        action_dim: int = ACTION_DIM,
        dropout:    float = 0.0,
    ) -> None:
        super().__init__()

        self.fused_dim  = fused_dim
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.LayerNorm(fused_dim),                      # stabilise fused input
            nn.Linear(fused_dim, hidden_dim),             # expand
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),            # project to actions
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, fused_cls: torch.Tensor) -> torch.Tensor:
        """
        fused_cls : (B, fused_dim)  — output of VisionLanguageFusion
        returns   : (B, action_dim) — predicted action deltas
          [dx, dy, dz, d_roll, d_pitch, d_yaw, d_gripper]
        """
        return self.net(fused_cls)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# FACTORY FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def build_action_head(
    config:     str   = "tiny",
    action_dim: int   = ACTION_DIM,
    dropout:    float = 0.0,
) -> ActionHead:
    """
    Convenience constructor.  Config must match the fusion module config used.

    Examples
    ────────
    >>> head = build_action_head("small")
    >>> action = head(fused_cls)   # (B, 7)
    """
    cfg = ACTION_HEAD_CONFIGS[config]
    return ActionHead(
        action_dim=action_dim,
        dropout=dropout,
        **cfg,
    )


# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE SMOKE TEST
# Wires together all 4 modules: VisionEncoder → LanguageEncoder → Fusion → ActionHead
# Usage:  python -m src.models.action_head
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from src.models.vision_encoder import build_vision_encoder
    from src.models.language_encoder import build_language_encoder
    from src.models.fusion import build_fusion_module

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[action_head test — full pipeline]  device = {device}")
    print("=" * 60)

    for config_name in ACTION_HEAD_CONFIGS:
        # ── Build all 4 modules ──────────────────────────────────────────
        vision   = build_vision_encoder(config_name).to(device)
        language = build_language_encoder(config_name).to(device)
        fusion   = build_fusion_module(config_name, num_fusion_layers=2).to(device)
        head     = build_action_head(config_name).to(device)

        # ── Fake inputs ──────────────────────────────────────────────────
        B = 4
        images    = torch.rand(B, 3, 84, 84, device=device)          # RGB camera frames
        token_ids = torch.randint(0, 2048, (B, 32), device=device)   # tokenised instruction

        # ── Full forward pass ────────────────────────────────────────────
        v_cls, v_tokens = vision(images)
        l_cls, l_tokens = language(token_ids)
        fused_cls, _    = fusion(v_cls, v_tokens, l_cls, l_tokens)
        action          = head(fused_cls)

        # ── Print shape summary ──────────────────────────────────────────
        total_params = (vision.num_parameters() + language.num_parameters() +
                        fusion.num_parameters() + head.num_parameters())
        print(f"\n  Config: {config_name}")
        print(f"    image input           : {list(images.shape)}")
        print(f"    token_ids input       : {list(token_ids.shape)}")
        print(f"    vision cls/patches    : {list(v_cls.shape)} / {list(v_tokens.shape)}")
        print(f"    language cls/tokens   : {list(l_cls.shape)} / {list(l_tokens.shape)}")
        print(f"    fused_cls             : {list(fused_cls.shape)}")
        print(f"    action output         : {list(action.shape)}")
        print(f"    action head params    : {head.num_parameters():,}")
        print(f"    TOTAL pipeline params : {total_params:,}")

        # ── Verify gradient flows through the entire pipeline ────────────
        loss = action.mean()
        loss.backward()
        print(f"    Backward pass (full)  : OK")

    print("\n[action_head test]  All configs passed.")
    print("\nAction output interpretation:")
    print("  index 0: dx        — Cartesian x delta (metres)")
    print("  index 1: dy        — Cartesian y delta (metres)")
    print("  index 2: dz        — Cartesian z delta (metres)")
    print("  index 3: d_roll    — roll delta (radians)")
    print("  index 4: d_pitch   — pitch delta (radians)")
    print("  index 5: d_yaw     — yaw delta (radians)")
    print("  index 6: d_gripper — gripper openness delta (−1=close, +1=open)")
    sys.exit(0)
