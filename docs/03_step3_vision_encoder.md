# Step 3: Vision Encoder From Scratch

## Goal
Build the vision backbone that converts an RGB robot image into learned visual tokens and a global image embedding.

## What was created
- `src/models/vision_encoder.py`

## Why this step matters
A VLA model cannot act from language alone. It must first convert images into a representation that the rest of the model can reason over.

This step builds that representation module from scratch using a small Vision Transformer (ViT).

## High-level flow
Input image:
- `(B, 3, 84, 84)`

Processing pipeline:
1. split image into `7x7` patches
2. project each patch into an embedding vector
3. prepend a learnable `CLS` token
4. add positional embeddings
5. run transformer blocks with self-attention
6. output:
- `cls_token`: `(B, embed_dim)`
- `patch_tokens`: `(B, 144, embed_dim)`

## Components in the file
- `PatchEmbedding`
- `MultiHeadSelfAttention`
- `MLP`
- `TransformerBlock`
- `VisionEncoder`
- `build_vision_encoder`

## Why patches
For `84x84` images with `patch_size=7`:
- `84 / 7 = 12`
- `12 x 12 = 144` image patches

The transformer treats these 144 patches like a token sequence.

## Why the `CLS` token exists
The `CLS` token is a learnable summary token. After attention layers, it gathers information from all patches and becomes a compact global image representation.

We will later use it for image-text contrastive learning.

## Why patch tokens are also returned
The full patch sequence keeps spatial detail. That will matter later when language tokens attend to visual regions during multimodal fusion.

## Significance of each layer (beginner-friendly)
Think of the ViT as a pipeline where each layer solves one specific problem.

### 1) PatchEmbedding
Problem it solves:
- Raw images have too many pixels to process directly as tokens.

What it does:
- Splits the image into non-overlapping `7x7` patches.
- Uses one `Conv2d(kernel=7, stride=7)` to convert each patch into a vector.

Why it is important:
- Reduces compute drastically.
- Creates a token sequence the Transformer can process.

Shape effect:
- `(B, 3, 84, 84)` -> `(B, 144, embed_dim)`

### 2) CLS token (learnable)
Problem it solves:
- We need one compact representation for the entire image.

What it does:
- Adds one extra token in front of patch tokens.
- During attention, this token reads information from all patches.

Why it is important:
- Gives a single vector summary for image-text alignment later.

Shape effect:
- `(B, 144, embed_dim)` -> `(B, 145, embed_dim)`

### 3) Positional embedding (learnable)
Problem it solves:
- Self-attention alone does not know patch order or location.

What it does:
- Adds a learned positional vector to each token index.

Why it is important:
- Encodes where each patch came from in the image.
- Without this, left/right/top/bottom would be ambiguous.

Shape effect:
- shape unchanged: `(B, 145, embed_dim)`

### 4) LayerNorm (inside each TransformerBlock)
Problem it solves:
- Activations can drift in scale during deep training.

What it does:
- Normalizes token features before attention and before MLP.

Why it is important:
- Stabilizes training.
- Enables deeper models and smoother optimization.

Shape effect:
- shape unchanged: `(B, 145, embed_dim)`

### 5) MultiHeadSelfAttention
Problem it solves:
- A patch needs context from other patches to understand objects/scenes.

What it does:
- Computes Q/K/V projections.
- For each token, mixes information from all other tokens.
- Multiple heads learn different relation patterns.

Why it is important:
- Gives global context in every block.
- Helps learn long-range dependencies (for example: arm and object far apart).

Shape effect:
- shape unchanged: `(B, 145, embed_dim)`

### 6) Residual connection after attention
Problem it solves:
- Deep nets can lose information and gradients.

What it does:
- Adds input back to attention output: `x = x + attn(...)`.

Why it is important:
- Preserves useful features.
- Improves gradient flow and convergence.

Shape effect:
- shape unchanged: `(B, 145, embed_dim)`

### 7) MLP (feed-forward network)
Problem it solves:
- Attention mixes tokens, but we also need per-token nonlinear transformation.

What it does:
- `Linear -> GELU -> Linear` with expansion ratio (typically `4x`).

Why it is important:
- Increases feature expressiveness.
- Lets each token compute richer local transformations after global mixing.

Shape effect:
- shape unchanged: `(B, 145, embed_dim)`

### 8) Residual connection after MLP
Problem it solves:
- Same stability issue as attention branch.

What it does:
- Adds skip path: `x = x + mlp(...)`.

Why it is important:
- Keeps optimization stable in stacked blocks.

Shape effect:
- shape unchanged: `(B, 145, embed_dim)`

### 9) Final LayerNorm
Problem it solves:
- Output scale consistency before downstream usage.

What it does:
- One final normalization across token features.

Why it is important:
- Makes output embeddings more stable for downstream heads/losses.

Shape effect:
- shape unchanged: `(B, 145, embed_dim)`

### 10) Output split: CLS token and patch tokens
Problem it solves:
- Different downstream modules need different granularity.

What it does:
- `cls_token = x[:, 0]`
- `patch_tokens = x[:, 1:]`

Why it is important:
- `cls_token` gives global representation.
- `patch_tokens` preserve spatial detail for fusion/cross-attention.

Shape effect:
- `cls_token`: `(B, embed_dim)`
- `patch_tokens`: `(B, 144, embed_dim)`

## One-line mental model
Each Transformer block does two things repeatedly:
1. mix information across tokens (attention)
2. enrich each token nonlinearly (MLP)

Residual paths and normalization make this trainable at depth.

## Available configs
The file defines three model sizes:
- `tiny`
- `small`
- `base`

Observed parameter counts from the self-test:
- `tiny`: `830,976`
- `small`: `4,814,336`
- `base`: `25,370,624`

## How to run
```bash
conda activate vla
cd <repo-root>
python -m src.models.vision_encoder
```

## What the self-test checks
- model builds for all three configs
- forward pass works
- output tensor shapes are correct
- backward pass works

## Expected output shapes
For `tiny` config and batch size `4`:
- input: `[4, 3, 84, 84]`
- cls output: `[4, 128]`
- patch output: `[4, 144, 128]`

## What you learned
- how a Vision Transformer tokenizes images
- why self-attention is useful for global visual reasoning
- how residual connections and layer norm stabilize training
- the difference between a global embedding and spatial tokens

## Output of this step
A working vision backbone that will later be paired with a language encoder and fusion module.
