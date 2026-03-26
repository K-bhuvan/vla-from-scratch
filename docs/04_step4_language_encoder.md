# Baby Step 4: Language Encoder — Text Transformer from Scratch

## Overview

The **Language Encoder** is a Transformer-based text encoder that converts robot instructions into fixed-size embeddings. Just like the Vision Encoder (Step 3) converts 84×84 RGB images into a representation, the Language Encoder converts tokenized instruction strings like *"pick the cube"* or *"place the object in the box"* into a compact vector.

### Output Contract

The Language Encoder returns two tensors:

1. **`cls_token : (B, embed_dim)`** — A single global representation of the entire instruction
   - Used for **contrastive loss in pre-training** (align image embeddings with instruction embeddings)
   - One vector per instruction, easy to compare with one vector per image

2. **`token_tokens : (B, seq_len, embed_dim)`** — Per-token semantic features  
   - Used for **cross-attention fusion in Step 5** (spatial fine-grained alignment)
   - Preserves semantic information at sub-instruction level
   - Example: which tokens refer to "the target object" vs. "the action"

---

## Architecture: 5 Building Blocks

### 1. TokenEmbedding — Convert Token IDs to Vectors

**Problem:** 
The model receives token IDs (integers 0-2047). Neural networks can't directly process integers — they need vectors.

**What it does:**
```python
self.embed = nn.Embedding(vocab_size=2048, embed_dim=128)
token_vectors = embed(token_ids)  # (B, seq_len) → (B, seq_len, embed_dim)
```

Each token ID maps to a **learned vector** of size `embed_dim`. Think of it as a giant lookup table:
- Token ID 5 (e.g., "pick") → 128-dimensional vector
- Token ID 12 (e.g., "the") → 128-dimensional vector  
- Token ID 78 (e.g., "cube") → 128-dimensional vector

**Why important:**
- Tokens with similar meanings will learn similar embeddings through training
- The distance between "pick" and "grasp" in embedding space will be small
- The distance between "pick" and "blue" will be large
- This is how the model learns semantic relationships in language

**Shape effect:**
- Input: `(B, seq_len)` — integers in range [0, vocab_size)
- Output: `(B, seq_len, embed_dim)` — dense vectors

---

### 2. CLS Token — Learnable Global Summary Vector

**Problem:**  
After the Transformer processes all tokens, we need a single vector to represent the *entire instruction*. We could take the average of all token vectors, but averaging is crude — we lose the ability to decide which tokens are important.

**What it does:**
```python
self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # learned vector
# In forward pass:
cls = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
x = torch.cat([cls, x], dim=1)  # prepend to token sequence → (B, seq_len+1, embed_dim)
```

The CLS token is a learnable parameter (initially random) that gets prepended to the token sequence. Through self-attention, it aggregates information from *all other tokens*.

**Intuition:**  
- Think of the CLS token as a "[SUMMARY]" placeholder that sits at position 0
- During attention, CLS can look at all tokens ("pick", "the", "cube", etc.)
- All tokens can broadcast information back to CLS
- By the end of all Transformer layers, CLS contains a distilled summary of the instruction

**Why important:**  
- Better than averaging because the model learns *what is important* to attend to
- Different sentence meanings → different CLS vectors (via trained attention)
- Similar instructions → similar CLS vectors (useful for contrastive loss later)

**Shape effect:**  
- Prepends one vector to the sequence
- Sequence length: `seq_len` → `seq_len + 1`

---

### 3. Positional Embedding — Teach Order Matters

**Problem:**  
Transformers are **permutation-invariant** — they treat the sequence as an unordered set. The word order in "pick the cube" vs. "cube the pick" would look identical without additional context. But word order matters a lot!

**What it does:**
```python
self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len + 1, embed_dim))
# In forward pass:
x = x + self.pos_embed[:, :seq_len + 1, :]  # add position information
```

Each position in the sequence gets a **learned embedding**:
- Position 0 (CLS token) → position vector P₀
- Position 1 (first token, "pick") → position vector P₁  
- Position 2 (second token, "the") → position vector P₂
- ... and so on

**Intuition:**  
- The position embedding is added to the token embedding
- This tells the model "this token is at position 2 in the sequence"
- Relative distances between positions help the model learn temporal structure

**Why learned (not fixed)?**  
- Learned positional embeddings adapt to the task
- The model can learn that "tokens close together" are more related
- Common in modern models (BERT, GPT, Vision Transformer all use learned positions)

**Shape effect:**  
- Doesn't change sequence length
- Just adds structured information to the embeddings

---

### 4. MultiHeadSelfAttention — Every Token Looks At Every Other Token

**Problem:**  
In "pick the red cube", the model needs to know that:
- "pick" is the ACTION
- "red" and "cube" both describe the OBJECT  
- "red cube" is a compound noun, not two separate concepts
- The action "pick" applies to the object "red cube"

A simple token-by-token MLP can't capture these relationships across the sequence.

**What it does:**
For each position in the sequence, compute **attention scores** to all other positions:

```
Q_i = query of token at position i  ("what am I looking for?")
K_j = key of token at position j     ("what do I offer?")
V_j = value of token at position j   ("what information do I carry?")

attention_score(i, j) = softmax( Q_i · K_j / √(head_dim) )
output_i = weighted_sum over all j of V_j
```

**Multi-head intuition:**  
- Head 1: Learns grammatical relationships (adjective-noun links like "red" ↔ "cube")
- Head 2: Learns semantic roles (action-object links like "pick" ↔ "cube")
- Head 3: Learns coreference (which pronouns refer to which entities)
- ... etc (8 heads in "small" config)

All heads run in parallel, then their output are concatenated and projected.

**Why important:**  
- Global context on every layer (unlike RNN which is sequential)
- Multiple heads learn different types of relationships
- Scales to long sequences (modern instructions can be "pick up the object near the blue cube on the left side of the table")

**Shape effect:**  
- Doesn't change shape: `(B, seq_len+1, embed_dim)` → `(B, seq_len+1, embed_dim)`
- But dramatically improves the *quality* of the vectors

---

### 5. MLP (Feed-Forward) — Per-Token Refinement

**Problem:**  
After attention mixes information across tokens, each token needs to refine its representation in its own "thinking space". Attention is good for *communication*; MLP is good for *individuation*.

**What it does:**
```python
hidden_dim = embed_dim * 4  # e.g., 128 * 4 = 512
MLP = Linear(embed_dim → 512) → GELU → Linear(512 → embed_dim)
```

For each token independently:
1. Expand to a higher dimension (512) — more expressivity
2. Apply GELU activation — smooth non-linearity
3. Project back to embed_dim — recombine features

**Intuition:**  
- Think of it as each token having a private workspace
- Token can combine mixed-in information from attention in novel ways
- Different tokens can learn different transformations (all using the same network, but with different inputs)

**Why important:**  
- Attention handles inter-token communication
- MLP handles intra-token feature transformation
- Together they form the fundamental Transformer block

**Shape effect:**  
- Doesn't change shape: `(B, seq_len+1, embed_dim)` → `(B, seq_len+1, embed_dim)`

---

### 6. LayerNorm — Stabilize Training

**Problem:**  
As information flows through multiple layers of attention and MLP, the activation magnitudes can grow or shrink unpredictably. This makes training unstable (vanishing/exploding gradients).

**What it does:**
```python
self.norm1 = nn.LayerNorm(embed_dim)  # before attention
self.norm2 = nn.LayerNorm(embed_dim)  # before MLP

x = x + attention(norm1(x))  # Pre-norm variant (modern standard)
x = x + mlp(norm2(x))
```

For each token independently, normalize to mean 0, std 1 within the embedding dimension. Then apply learned scale and shift parameters.

**Why important:**  
- Keeps activation ranges stable across layers
- Allows higher learning rates (faster training)
- "Pre-norm" (normalizing *before* attention/MLP) is more stable than "post-norm"

**Shape effect:**  
- Doesn't change shape
- But makes all other operations more numerically stable

---

### 7. Residual Connections — Gradient Flow Highway

**Problem:**  
In a 6-layer Transformer, gradients from the loss have to backprop through 12 matrix multiplications (2 per block: attention + MLP). This can cause vanishing gradients in early layers.

**What it does:**
```python
x = x + attention(norm(x))  # ← the "+ x" part
x = x + mlp(norm(x))        # ← the "+ x" part
```

The output is added to the input. Gradients can flow directly from output back to input via the **+ operation**, which has gradient 1.0 (no multiplication that shrinks values).

**Intuition:**  
- Information has two paths: one through attention/MLP, one direct shortcut
- Both paths exist simultaneously
- Gradients can shortcut through the skip connection
- Information doesn't get forced into a bottleneck

**Why important:**  
- Makes deep networks trainable
- Allows stable learning across 6-8 layers in "small"/"base" configs
- ResNet proved this for CNNs; Transformers use the same principle

**Shape effect:**  
- Doesn't change shape
- But dramatically improves trainability

---

### 8. Output: CLS Token Extraction

**Problem:**  
The Transformer outputs `(B, seq_len+1, embed_dim)` — one vector per token plus CLS. But we need a single instruction representation.

**What it does:**
```python
x = output from all transformer layers  # (B, seq_len+1, embed_dim)
cls_token = x[:, 0]                     # (B, embed_dim) — just the CLS position
token_tokens = x[:, 1:]                 # (B, seq_len, embed_dim) — everything else
```

Simply extract the CLS token (position 0) as the global instruction representation, and keep all other tokens for fine-grained cross-attention in Step 5.

**Why this works:**  
- CLS has attended to all tokens and been attended to by all tokens
- It truly represents the whole instruction
- Token-level features are preserved for spatial reasoning in the vision-language fusion

---

## Configuration Variants

The LanguageEncoder comes in 3 pre-built sizes:

| Config | embed_dim | num_heads | num_layers | vocab_size | max_seq_len | ~Params |
|--------|-----------|-----------|-----------|-----------|------------|---------|
| **tiny** | 128 | 4 | 4 | 2048 | 128 | 0.5 M |
| **small** | 256 | 8 | 6 | 2048 | 128 | 2.8 M |
| **base** | 512 | 8 | 8 | 2048 | 128 | 10 M |

**Use "tiny" for MVP**, "small" for better capacity, "base" for maximum expressivity.

---

## Comparison with Vision Encoder (Step 3)

| Aspect | Vision Encoder | Language Encoder |
|--------|---|---|
| **Input** | 84×84 RGB image | Tokenized instruction |
| **Input type** | Continuous pixels (0-1) | Discrete token IDs (0-2047) |
| **Embedding layer** | Conv2d with patch_size=7 | Embedding table (vocab_size × embed_dim) |
| **Sequence length** | 144 patches (12×12 grid) | max_seq_len=128 tokens |
| **Prepended token** | CLS token | CLS token |
| **Transformer blocks** | Same MultiHeadSelfAttention, MLP | Same MultiHeadSelfAttention, MLP |
| **Output** | cls_token + patch_tokens | cls_token + token_tokens |
| **embed_dim compatibility** | 128/256/512 | 128/256/512 (same for fusion!) |

Both use the **same Transformer architecture**, just with different input representations. This makes them easy to fuse in Step 5.

---

## Data Flow: A Concrete Example

**Instruction:** "pick the red cube"

**Step 1: Tokenization**  
(Assuming a simple vocabulary)
```
"pick" → token_id = 5
"the"  → token_id = 12
"red"  → token_id = 89
"cube" → token_id = 78
Padding to seq_len=128: [5, 12, 89, 78, 0, 0, ..., 0]
```

**Step 2: Token Embedding**  
```
5  → [0.15, -0.32, ..., 0.81]  (128-dim vector)
12 → [0.42, -0.01, ..., -0.55]
89 → [0.88, 0.15, ..., 0.22]
78 → [-0.33, 0.61, ..., 0.44]
```

**Step 3: Add CLS, Positional Embeddings**  
```
[CLS]          → [learned embed]  + [position 0 embed]
"pick" token   → [word embed]     + [position 1 embed]
"the" token    → [word embed]     + [position 2 embed]
"red" token    → [word embed]     + [position 3 embed]
"cube" token   → [word embed]     + [position 4 embed]
[PAD] ... [PAD] → [embed 0]       + [position 5-128 embed]
```

**Step 4: Self-Attention (First Block)**  
- CLS attends to "pick", "the", "red", "cube"
- "pick" attends to CLS and "the", "red", "cube"
- "red" attends to CLS, "pick", "the", "cube"
- ... and so on (global attention)

Result: Mixed tokens that have learned relationships

**Step 5: MLP (First Block)**  
- Each token individually refines its representation
- No cross-token communication, just per-token processing

**Step 6-7: Repeat** (blocks 2-4 for "tiny" config)
- Each block deepens understanding
- Semantic relationships solidify (e.g., "red" and "cube" bond together)

**Step 8: Final Output**  
```
CLS token: [-0.23, 0.56, ..., 0.15]  ← low-dim summary of the instruction
"pick" token: [0.41, 0.12, ..., -0.38]
"the" token:  [0.03, -0.67, ..., 0.22]
...
```

---

## How to Use

### Quick Start

```python
from src.models.language_encoder import build_language_encoder

# Load a pre-built configuration
encoder = build_language_encoder("small")

# Or specify custom vocab/seq_len
encoder = build_language_encoder(
    config="tiny",
    vocab_size=2048,
    max_seq_len=128,
)

# Forward pass with tokenized instructions
token_ids = torch.randint(0, 2048, (batch_size, 128))
cls_token, token_tokens = encoder(token_ids)
# cls_token: (batch_size, 256)
# token_tokens: (batch_size, 128, 256)
```

### Self-Test

Verify the encoder works end-to-end:

```bash
python -m src.models.language_encoder
```

Expected output:
```
[language_encoder test]  device = cpu
============================================================

  Config: tiny
    Input shape         : [4, 32]
    CLS output shape    : [4, 128]
    Token output shape  : [4, 32, 128]
    Trainable params    : 514,816
    Backward pass       : OK
    
  ... (small and base configs) ...
    
[language_encoder test]  All configs passed.
```

---

## What Happens During Training?

1. **Token embeddings evolve** — "pick" and "grasp" grow closer, "pick" and "blue" grow farther
2. **Attention heads specialize**:
   - Head 1: Syntax (adjective-noun bonds)
   - Head 2: Semantics (action-object bonds)
   - Head 3: Coreference (pronoun resolution)
3. **CLS token accumulates meaning** — by layer 4, it's a rich summary
4. **Position embeddings tune** — the model learns relative position is more important than absolute

By the end of pre-training (contrastive loss over image-instruction pairs), the encoder understands language *and* its relationship to visual observations.

---

## Next Step: Vision-Language Fusion (Step 5)

The Vision Encoder (Step 3) and Language Encoder (Step 4) output embeddings with **matching `embed_dim`** (e.g., 256 for "small").

In Step 5, we'll:
1. Concatenate or cross-attend cls_token (image) with cls_token (language)
2. Use cross-attention between vision patch_tokens and language token_tokens
3. Produce a joint fused representation ready for action prediction

This design mirrors modern vision-language models (CLIP, LLaVA, RT-2).
