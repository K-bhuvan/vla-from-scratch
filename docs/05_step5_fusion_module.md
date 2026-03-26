# Baby Step 5: Fusion Module — Vision-Language Cross-Attention from Scratch

## Overview

The **Fusion Module** combines the outputs from the Vision Encoder (Step 3) and Language Encoder (Step 4) using **cross-attention**. 

Instead of simply concatenating embeddings, cross-attention lets:
- **Vision patches** ask "which language tokens describe me?" 
- **Language tokens** ask "which visual regions correspond to me?"

This creates a **fine-grained alignment**: each pixel region understands its linguistic context, and each word understands its visual grounding.

### Output Contract

The Fusion Module takes separate multimodal inputs and outputs jointly fused representations:

**Inputs:**
- `vision_cls : (B, embed_dim)` — global image representation from VisionEncoder
- `vision_tokens : (B, 144, embed_dim)` — spatial patch features from VisionEncoder  
- `language_cls : (B, embed_dim)` — global instruction representation from LanguageEncoder
- `language_tokens : (B, seq_len, embed_dim)` — semantic token features from LanguageEncoder

**Outputs:**
- `fused_cls : (B, 2 * embed_dim)` — global joint understanding (ready for action prediction)
- `fused_tokens : (B, 144, 2 * embed_dim)` — spatially-grounded language features

---

## Architecture: 2 Building Blocks

### 1. CrossAttention — Query One Modality, Attend to Another

**Problem:**  
Self-attention (Steps 3-4) lets tokens understand *within their own modality*. But we need *cross-modal understanding*: which parts of the image relate to which words?

**What it does:**

```python
# Vision patch: "I'm at position (row=5, col=3). What language describes me?"
Q = vision_patch @ W_Q_vision    # Query from vision

# Language tokens: "I offer 'pick', 'red', 'cube', etc."
K = language_tokens @ W_K_lang   # Keys from language
V = language_tokens @ W_V_lang   # Values from language

# Attention
# d_k = embed_dim / num_heads  (e.g. 256 / 8 = 32 for "small" config)
# Dividing by √d_k prevents dot products from growing too large and saturating softmax
attention_scores = softmax( Q @ K^T / √d_k )  # shape: (B, heads, 144 patches, N_lang tokens)
output = attention_scores @ V                   # Weighted blend of language embeddings
```

**Intuition:**  
- A vision patch doesn't just understand itself (self-attention did that)
- It now understands which words in the instruction are most relevant to it
- The patch at position (5, 3) might learn: "I'm mostly 'cube' and 'red'"
- The patch at position (0, 0) might learn: "I'm mostly 'table' and 'background'"

**Why different from self-attention:**

| Aspect | Self-Attention | Cross-Attention |
|--------|---|---|
| Q source | Same tokens | One modality |
| K source | Same tokens | Other modality |
| V source | Same tokens | Other modality |
| Use case | Intra-modality understanding | Cross-modal alignment |

**Shape effect:**
- Input: Q from one modality `(B, N_q, embed_dim)`, KV from another `(B, N_kv, embed_dim)`
- Output: `(B, N_q, embed_dim)` — query tokens enriched with key/value context

---

### 2. FusionBlock — Bidirectional Cross-Attention

**Problem:**  
One-directional attention is incomplete. Vision patches learn about language, but language tokens don't learn where they apply in the image.

**What it does:**

```
Vision Path:
  vision_patches → CrossAttention(Q=patches, KV=language) → language-grounded patches

Language Path:
  language_tokens → CrossAttention(Q=tokens, KV=patches) → spatially-grounded tokens

Both branches have residual connections and MLPs for refinement.
```

**Intuition:**  
- Direction 1 (V→L): "I'm a visual patch. Let me look at all language tokens and understand what's happening around me."
- Direction 2 (L→V): "I'm a language token. Let me look at all visual patches and understand where I apply in the image."
- Result: **Tight bidirectional alignment** — each modality understands the other.

**Architecture:**

```
┌──────────────────────────────────┐
│  FusionBlock                      │
├──────────────────────────────────┤
│  Vision Attention Flow:           │
│    vision_tokens                  │
│      + LayerNorm                  │
│      + CrossAttn(Q=v, KV=lang)    │
│      + residual                   │
│      + MLP                        │
│      + residual                   │
│    = fused_vision_tokens          │
│                                   │
│  Language Attention Flow:         │
│    language_tokens                │
│      + LayerNorm                  │
│      + CrossAttn(Q=lang, KV=v)    │
│      + residual                   │
│      + MLP                        │
│      + residual                   │
│    = fused_language_tokens        │
└──────────────────────────────────┘
```

**Why bidirectional:**
- Unidirectional: vision learns about language, but language doesn't adapt to vision
- Bidirectional: both modalities learn about each other iteratively
- Result: More balanced alignment; both modalities grow closer in semantic space

---

## The 3-Step Fusion Process

### Step 1: Bidirectional Cross-Attention (via Fusion Blocks)

**Input:**
- Vision patches: `(B, 144, embed_dim)` — spatial, what each region looks like
- Language tokens: `(B, seq_len, embed_dim)` — semantic, what the instruction says

**Block 1:**
1. Patches attend to tokens → patches learn instruction context
2. Tokens attend to patches → tokens learn visual grounding
3. Both run through residual MLPs

**Block 2** (if num_fusion_layers=2):
- Repeat with already-aligned representations for deeper fusion
- Patches now have language context from Block 1, and learn even more
- Tokens now have spatial grounding from Block 1, and learn even more

**Output after N blocks:**
- Vision_tokens: `(B, 144, embed_dim)` — each patch now "knows" the full instruction
- Language_tokens: `(B, seq_len, embed_dim)` — each word now "knows" the full image

---

### Step 2: Normalize Outputs

```python
vision_tokens = norm_vision_out(vision_tokens)
language_tokens = norm_language_out(language_tokens)
```

Stabilizes activations before concatenation.

---

### Step 3: Concatenate Modalities

**For the global representation (CLS):**
```python
fused_cls = concat(vision_cls, language_cls)
# (B, embed_dim) + (B, embed_dim) → (B, 2 * embed_dim)
```

One vector per image-instruction pair, combining both understandings.

**For spatial tokens:**
```python
fused_tokens = concat(vision_tokens, language_tokens[:, :144])
# (B, 144, embed_dim) + (B, 144, embed_dim) → (B, 144, 2 * embed_dim)
```

Note: Only use the first 144 language tokens (matching vision patch count) to keep the token count aligned.

---

## Configuration Variants

The VisionLanguageFusion comes in 3 pre-built sizes:

| Config | embed_dim | num_heads | num_fusion_layers | ~Params |
|--------|-----------|-----------|-----------|---------|
| **tiny** | 128 | 4 | 2 | 1.2 M |
| **small** | 256 | 8 | 2 | 6.8 M |
| **base** | 512 | 8 | 2 | 27 M |

**Default: num_fusion_layers=2**

- num_fusion_layers=1: Faster, lighter (~60% params), adequate for MVP
- num_fusion_layers=2: **Recommended** — better alignment, minimal overhead
- num_fusion_layers=3: Slower, fuller alignment, for when overfitting is not a concern

---

## Data Flow: A Concrete Example

**Setup:**
- Instruction: "pick the red cube"
- Image: Table with red cube, blue cylinder, green sphere
- Using "small" config (embed_dim=256)

**Input to Fusion:**
```
vision_cls  : (1, 256)       — global image summary
vision_tokens : (1, 144, 256) — spatial features:
                                  patches[5, 3]  ← red region
                                  patches[2, 5]  ← cube region
                                  patches[1, 1]  ← background
                                  … (144 total patches)

language_cls  : (1, 256)     — global instruction understanding
language_tokens : (1, 32, 256) — semantic tokens:
                                   tok[0] = "pick"     (action)
                                   tok[1] = "the"      (determiner)
                                   tok[2] = "red"      (color)
                                   tok[3] = "cube"     (object)
                                   tok[4:] = padding
```

**Block 1 — Bidirectional Cross-Attention:**

**Sub-step 1a (Vision → Language):**
```
For each vision patch:
  patch[5, 3] (red region) asks:
    "Which language tokens describe me?"
    Attention distribution:
      tok[2] "red"       ← high (97%)
      tok[3] "cube"      ← medium (1%)
      tok[0] "pick"      ← low (1%)
      tok[1] "the"       ← low (1%)
    Result: patch[5, 3] gets a blend of embeddings weighted by this distribution

  patch[2, 5] (cube region) asks:
    Attention distribution:
      tok[3] "cube"      ← high (95%)
      tok[2] "red"       ← medium (2%)
      tok[0] "pick"      ← low (2%)
      tok[1] "the"       ← low (1%)
    Result: patch[2, 5] gets a blend weighted differently

  patch[1, 1] (background) asks:
    Attention distribution:
      tok[1] "the"       ← high (50%)
      tok[0] "pick"      ← high (30%)
      tok[2], tok[3]     ← low (20%)
    Result: background learns this is part of the action's context
```

**Sub-step 1b (Language → Vision):**
```
For each language token:
  tok[0] "pick" (action verb) asks:
    "Which visual regions apply to me?"
    Might learn: "I apply to all regions that are involved in grasping"
    Attends to patches showing arm, gripper position, etc.

  tok[2] "red" (color) asks:
    "Which visual regions have this property?"
    Attends heavily to patches[5, 3] (red cube region)
    Attends lightly to other regions

  tok[3] "cube" (shape) asks:
    "Which visual regions match this shape?"
    Attends to patches showing cubic structure at patches[2, 5]
    Attends less to spherical/cylindrical regions
```

**After Block 1:**
- Vision patches: enriched with instruction semantics
- Language tokens: grounded in specific image regions
- Both modalities now have cross-modal context

**Block 2 — Deeper Fusion (if num_fusion_layers=2):**
- Same process, but now with already-aligned representations
- Patches refine their understanding of how "red", "cube", "pick" interact
- Tokens refine their understanding of where those concepts appear together

**After All Blocks:**
```
vision_tokens[5, 3]  : (1, 256) — raw patch + language context ("red" + "cube" + "pick")
language_tokens[2]   : (1, 256) — "red" token + spatial context (appears at patch[5,3])
```

**Final Concatenation:**
```
fused_cls : concat(vision_cls, language_cls)           → (1, 512)
fused_tokens : concat(vision_tokens, language_tokens)  → (1, 144, 512)
```

**Ready for Action Head:**
- `fused_cls`: Contains full multimodal understanding
  - What the image shows (from vision branch)
  - What the instruction says (from language branch)
  - How they relate (from fusion blocks)
- `fused_tokens`: Spatial features grounded in instruction semantics
  - Each patch knows which language concepts apply to it
  - Useful for spatial attention in imitation learning

---

## Why Cross-Attention Over Alternatives?

### Option 1: Simple Concatenation ❌

```python
fused = concat(vision_cls, language_cls)  # (B, 2 * embed_dim)
```

**Pros:** Fast, no parameters
**Cons:** No learned alignment; model must learn relationships from scratch in the action head

### Option 2: Self-Attention on Concatenated Features ❌

```python
concatenated = concat(vision_patches, language_tokens)
fused = self_attention(concatenated)
```

**Pros:** Learns some alignment
**Cons:** Treats all sequences equally; doesn't leverage modality-specific structures (spatial vs. semantic)

### Option 3: Cross-Attention (This Approach) ✅

```python
# Vision → Language
vision_tokens_fused = cross_attention(Q=vision_tokens, KV=language_tokens)

# Language → Vision
language_tokens_fused = cross_attention(Q=language_tokens, KV=vision_tokens)
```

**Pros:**
- Explicitly designed for cross-modal fusion
- Maintains modality-specific token structures
- Bidirectional: both modalities learn from each other
- Proven effective in CLIP, BLIP, LLaVA, RT-2, OpenVLA

**Cons:** Slightly more parameters than simple concat, but worth it for the alignment quality

---

## Comparison with Vision & Language Encoders

| Aspect | Vision Encoder | Language Encoder | Fusion Module |
|--------|---|---|---|
| **Input** | Image (84×84) | Tokens | Both modalities |
| **Processing** | Self-Attention (patches look at patches) | Self-Attention (tokens look at tokens) | Cross-Attention (patches ↔ tokens) |
| **Output dim** | embed_dim | embed_dim | **2 × embed_dim** (concatenated) |
| **Job** | Image → embedding | Text → embedding | Embedding + Embedding → Joint Embedding |

---

## How to Use

### Quick Start

```python
from src.models.fusion import build_fusion_module
from src.models.vision_encoder import build_vision_encoder
from src.models.language_encoder import build_language_encoder

# Build all three modules
vision = build_vision_encoder("small")
language = build_language_encoder("small")
fusion = build_fusion_module("small", num_fusion_layers=2)

# Forward pass
images = torch.randn(4, 3, 84, 84)  # 4 images
v_cls, v_tokens = vision(images)    # (4, 256) + (4, 144, 256)

token_ids = torch.randint(0, 2048, (4, 32))  # 4 instructions
l_cls, l_tokens = language(token_ids)  # (4, 256) + (4, 32, 256)

# Fuse
fused_cls, fused_tokens = fusion(v_cls, v_tokens, l_cls, l_tokens)
# fused_cls: (4, 512)         — ready for action head
# fused_tokens: (4, 144, 512) — spatial + semantic features
```

### Self-Test

Verify the fusion module works end-to-end:

```bash
python -m src.models.fusion
```

Expected output:
```
Config: tiny
  Input vision_cls      : [4, 128]
  Input vision_tokens   : [4, 144, 128]
  Input language_cls    : [4, 128]
  Input language_tokens : [4, 32, 128]
  Output fused_cls      : [4, 256]
  Output fused_tokens   : [4, 144, 256]
  Trainable params      : 1,210,368
  Backward pass         : OK
```

---

## Parameter Count & Memory

The fusion module is lightweight:

**Tiny config (embed_dim=128, 2 blocks):**
- Cross-Attention layers: 2 × 2 blocks × (Q, KV, Out projections) ≈ 0.4 M
- MLPs: 2 × 2 blocks × (up + down) ≈ 0.5 M
- LayerNorms: ≈ 0.02 M
- **Total: ~0.73 M trainable parameters**

**Small config (embed_dim=256, 2 blocks):**
- **Total: ~3.8 M trainable parameters**

**Base config (embed_dim=512, 2 blocks):**
- **Total: ~15 M trainable parameters**

Memory is dominated by:
1. Attention computations (quadratic in sequence length, but cross-attention uses different lengths)
2. Vision patches: 144 × 256 = 36K per batch
3. Language tokens: 32 × 256 = 8K per batch

---

## Next Step: Action Head (Step 6)

Now that we have a joint multimodal representation `fused_cls : (B, 2 * embed_dim)`, we need to turn it into robot actions.

The Action Head will be a simple MLP:
```
fused_cls → Linear → GELU → Linear → [7 action deltas]
```

This produces [dx, dy, dz, d_roll, d_pitch, d_yaw, d_gripper] — the robot's incremental control.

---

## Reading Order

1. **This document** — understand fusion architecture
2. **src/models/fusion.py** — read the code
3. **Run self-test** — `python -m src.models.fusion`
4. **Proceed to Step 6** — Action Head
