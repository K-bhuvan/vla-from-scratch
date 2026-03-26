# Baby Step 6: Action Head — Policy MLP from Scratch

## Overview

The **Action Head** is the final module in the VLA pipeline. It takes the fused multimodal representation from Step 5 and maps it to **7 continuous action deltas** — the robot's next move.

```
fused_cls (B, 2 * embed_dim)
      │
      ▼
  ActionHead (MLP)
      │
      ▼
action (B, 7)
  [dx, dy, dz, d_roll, d_pitch, d_yaw, d_gripper]
```

---

## What are the 7 Action Dimensions?

| Index | Name | Meaning | Unit |
|-------|------|---------|------|
| 0 | `dx` | End-effector x translation | metres per step |
| 1 | `dy` | End-effector y translation | metres per step |
| 2 | `dz` | End-effector z translation | metres per step |
| 3 | `d_roll` | Roll rotation delta | radians per step |
| 4 | `d_pitch` | Pitch rotation delta | radians per step |
| 5 | `d_yaw` | Yaw rotation delta | radians per step |
| 6 | `d_gripper` | Gripper openness delta | −1 = close, +1 = open |

These are **delta** (incremental) actions. The robot reads its current state, adds the delta, and moves to the new state. This is consistent with the data generated in Step 2 (`ACTION_DIM = 7`).

---

## Architecture

### Why an MLP (not another Transformer)?

At this point, `fused_cls` is already a fully contextualised vector:
- Vision branch understood the image
- Language branch understood the instruction
- Fusion aligned them via cross-attention

The only remaining task is **regression**: map a single vector to 7 numbers. MLPs are the right tool here — simple, fast, and expressive enough. A Transformer would add overhead with no benefit since there is no sequence to attend over. All major VLA models (RT-2, OpenVLA, etc.) use an MLP for the action regression head.

### Structure (2-layer MLP)

```
Input: fused_cls (B, fused_dim)
  │
  ├── LayerNorm(fused_dim)
  │     Normalises the concatenated vision+language vector.
  │     Needed because the two halves (vision_cls ∥ language_cls) may be
  │     at different scales early in training.
  │
  ├── Linear(fused_dim → hidden_dim)
  │     Expands to a larger feature space.
  │     hidden_dim = fused_dim * 2  (e.g., 512 → 1024 for "tiny")
  │
  ├── GELU()
  │     Smooth non-linearity. Better gradient flow than ReLU near zero.
  │
  ├── Dropout(p)
  │     Optional regularisation. Set to 0.0 for MVP.
  │
  └── Linear(hidden_dim → 7)
        Projects down to ACTION_DIM = 7.
        No output activation (see note below).

Output: action (B, 7)
```

### Why no output activation (no tanh)?

Three reasons:
1. **Training targets are already normalised** — action deltas in the dataset live roughly in [−1, 1] by design. The MSE loss naturally pushes the outputs there.
2. **tanh saturates** at ±1 and has near-zero gradients at the extremes, which slows training when the model needs to predict values close to ±1.
3. **Clipping is done outside the model** — at inference time, actions are clipped to safe workspace bounds inside the environment wrapper, not inside the neural network.

---

## Configuration Variants

The ActionHead configs are matched to the fusion module's output dimension:

| Config | fused_dim | hidden_dim | action_dim | ~Params |
|--------|-----------|------------|------------|---------|
| **tiny** | 256 | 512 | 7 | 133 K |
| **small** | 512 | 1024 | 7 | 533 K |
| **base** | 1024 | 2048 | 7 | 2.1 M |

`fused_dim = 2 × embed_dim` because the fusion module concatenates `vision_cls ∥ language_cls`.

---

## Full Pipeline Parameter Count

Calling `python -m src.models.action_head` tests all 4 modules wired together end-to-end:

| Config | Vision | Language | Fusion | ActionHead | **Total** |
|--------|--------|----------|--------|------------|-----------|
| tiny | 1.5 M | 1.1 M | 0.8 M | 0.13 M | **~2.8 M** |
| small | 8.4 M | 5.3 M | 3.2 M | 0.53 M | **~13.8 M** |
| base | 38 M | 26 M | 12.6 M | 2.1 M | **~66 M** |

For our MVP we target **"small"** (~14M params) — a good balance of capacity and GPU memory.

---

## Data Flow: All 4 Modules Together

```
image (B, 3, 84, 84)              token_ids (B, 32)
       │                                  │
       ▼                                  ▼
VisionEncoder (Step 3)          LanguageEncoder (Step 4)
  patch_size=7, 144 patches       vocab_size=2048
       │                                  │
  v_cls (B, E)                     l_cls (B, E)
  v_tokens (B, 144, E)             l_tokens (B, 32, E)
       │                                  │
       └──────────────────────────────────┘
                          │
                          ▼
              VisionLanguageFusion (Step 5)
               bidirectional cross-attention
                          │
               fused_cls (B, 2E)
               fused_tokens (B, 144, 2E)
                          │
                          ▼
                   ActionHead (Step 6)
                   LayerNorm → Linear → GELU → Linear
                          │
                    action (B, 7)
          [dx, dy, dz, d_roll, d_pitch, d_yaw, d_gripper]
```

---

## How to Use

### Quick Start

```python
from src.models.vision_encoder import build_vision_encoder
from src.models.language_encoder import build_language_encoder
from src.models.fusion import build_fusion_module
from src.models.action_head import build_action_head

# Build all 4 modules (use matching config string for all)
vision   = build_vision_encoder("small")
language = build_language_encoder("small")
fusion   = build_fusion_module("small", num_fusion_layers=2)
head     = build_action_head("small")

# Forward pass
images    = torch.rand(4, 3, 84, 84)           # 4 RGB camera frames
token_ids = torch.randint(0, 2048, (4, 32))    # 4 tokenised instructions

v_cls, v_tokens = vision(images)               # (4, 256) + (4, 144, 256)
l_cls, l_tokens = language(token_ids)          # (4, 256) + (4, 32, 256)
fused_cls, _    = fusion(v_cls, v_tokens, l_cls, l_tokens)  # (4, 512)
action          = head(fused_cls)              # (4, 7)
```

### Self-Test (full pipeline)

```bash
python -m src.models.action_head
```

Expected output:
```
Config: small
  image input           : [4, 3, 84, 84]
  token_ids input       : [4, 32]
  vision cls/patches    : [4, 256] / [4, 144, 256]
  language cls/tokens   : [4, 256] / [4, 32, 256]
  fused_cls             : [4, 512]
  action output         : [4, 7]
  action head params    : 533,511
  TOTAL pipeline params : 13,804,551
  Backward pass (full)  : OK
```

---

## What Happens During Training?

In **Stage B (SFT, Step 8)**, the action head is trained with MSE loss against expert actions from the dataset:

```python
pred_action = head(fused_cls)          # (B, 7)  — model prediction
gt_action   = batch["action"]          # (B, 7)  — expert action from dataset
loss = F.mse_loss(pred_action, gt_action)
```

The gradient flows backwards through:
```
ActionHead → Fusion → VisionEncoder + LanguageEncoder
```

All 4 modules update simultaneously — the whole pipeline learns together.

---

## Next Step: Stage A Pre-training (Step 7)

Before full SFT, we run **contrastive pre-training** to align vision and language embeddings. This teaches the model that:
- The image of a red cube pairs with the instruction "pick the red cube"
- The image of a green sphere pairs with "move the ball to the box"

Only `vision_encoder`, `language_encoder`, and `fused_cls` are involved here (not the action head yet). After pre-training, the model's representations are well-aligned, making SFT converge faster and generalise better.
