"""
src/models/vision_encoder.py  —  Baby Step 3: Vision Encoder (Tiny ViT from scratch)

════════════════════════════════════════════════════════════════════════════════
WHAT THIS FILE BUILDS
════════════════════════════════════════════════════════════════════════════════
A Vision Transformer (ViT) that turns a robot camera image into a fixed-size
vector of numbers — the "image embedding" the rest of the VLA model will use.

    Input  : (batch, 3, 84, 84)  — a batch of RGB camera frames
    Output : (batch, embed_dim)  — a compact vector representing what was seen

════════════════════════════════════════════════════════════════════════════════
WHY A TRANSFORMER AND NOT A CNN?
════════════════════════════════════════════════════════════════════════════════
CNNs (like ResNet) process images through sliding local filters.  They are
great but have a hard time attending to distant parts of an image at once.

Transformers treat an image as a SEQUENCE of patches and let every patch
attend to every other patch in a single layer.  This "global receptive field
on every layer" is why ViTs are dominant in modern vision-language models
(CLIP, LLaVA, RT-2, OpenVLA all use ViT backbones).

════════════════════════════════════════════════════════════════════════════════
THE 5 BUILDING BLOCKS (read in order)
════════════════════════════════════════════════════════════════════════════════
  1. PatchEmbedding        — split image into patches, project to embed_dim
  2. MultiHeadSelfAttention— the core attention mechanism
  3. MLP                   — 2-layer feed-forward network inside each block
  4. TransformerBlock      — LayerNorm → Attention → residual → LayerNorm → MLP → residual
  5. VisionEncoder         — stacks the above, adds CLS token + positional encoding

════════════════════════════════════════════════════════════════════════════════
DATA FLOW DIAGRAM
════════════════════════════════════════════════════════════════════════════════

  Image (B, 3, 84, 84)
        │
        ▼
  ┌─────────────────────┐
  │  PatchEmbedding     │  Split into 12×12=144 patches of size 7×7×3,
  │  (patch_size=7)     │  flatten each patch → project to embed_dim vector.
  └─────────────────────┘
        │  (B, 144, embed_dim)
        ▼
  Prepend CLS token     →  (B, 145, embed_dim)
        │
        ▼
  Add positional embeddings  (learned, shape (1, 145, embed_dim))
        │
        ▼
  ┌─────────────────────┐
  │  Transformer Block  │  ×N
  │  ─────────────────  │
  │  LayerNorm          │
  │  → MultiHead Attn   │  every patch looks at every other patch
  │  + residual          │
  │  LayerNorm          │
  │  → MLP              │  per-patch feature transformation
  │  + residual          │
  └─────────────────────┘
        │  (B, 145, embed_dim)
        ▼
  Final LayerNorm
        │
        ▼
  Take CLS token [index 0]  →  (B, embed_dim)  ← THE IMAGE REPRESENTATION
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange


# ─────────────────────────────────────────────────────────────────────────────
# PRE-DEFINED SIZE CONFIGURATIONS
#
# "tiny" is what we use for the MVP.  The exact same VisionEncoder class works
# for all three — just change the numbers.
# ─────────────────────────────────────────────────────────────────────────────

VIT_CONFIGS: dict[str, dict] = {
    #           embed_dim  heads  layers  mlp_ratio   approx params (84×84, patch=7)
    "tiny":  dict(embed_dim=128, num_heads=4, num_layers=4,  mlp_ratio=4.0),  # ~1.5 M
    "small": dict(embed_dim=256, num_heads=8, num_layers=6,  mlp_ratio=4.0),  # ~8.4 M
    "base":  dict(embed_dim=512, num_heads=8, num_layers=8,  mlp_ratio=4.0),  # ~38 M
}


# ─────────────────────────────────────────────────────────────────────────────
# BUILDING BLOCK 1: PATCH EMBEDDING
# ─────────────────────────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """
    Split an image into non-overlapping patches and linearly project each one.

    INTUITION
    ─────────
    Think of the image as a newspaper page.  Instead of reading letter-by-letter
    (pixels), we cut it into 7×7-pixel tiles (patches) and describe each tile
    with a single embedding vector.  The model then works with those 144 tiles
    as if they were words in a sentence.

    WHY CONV2D FOR PATCHING?
    ─────────────────────────
    A Conv2d with kernel_size=patch_size and stride=patch_size is mathematically
    equivalent to: flatten each patch → linear projection.  The Conv2d version
    is faster because it is fused in one GPU kernel.

    PARAMETER COUNT (tiny config, 84×84 image, patch 7×7)
    ──────────────────────────────────────────────────────
      Conv2d(3, 128, kernel=7, stride=7)
      weights: 3 × 7 × 7 × 128 = 18,816
      bias:    128
      total:   18,944  ← tiny
    """

    def __init__(
        self,
        image_size:  int = 84,
        patch_size:  int = 7,
        in_channels: int = 3,
        embed_dim:   int = 128,
    ) -> None:
        super().__init__()
        assert image_size % patch_size == 0, (
            f"image_size ({image_size}) must be divisible by patch_size ({patch_size})"
        )

        self.patch_size  = patch_size
        self.num_patches = (image_size // patch_size) ** 2  # 12×12 = 144

        # Single Conv2d does patching + projection in one step
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, H, W)
        returns : (B, num_patches, embed_dim)
        """
        # After conv: (B, embed_dim, H/patch, W/patch)  e.g. (B, 128, 12, 12)
        x = self.proj(x)

        # Flatten the spatial grid into a sequence of patch tokens.
        # rearrange makes the intent explicit:
        #   b = batch, e = embed_dim, h = patch rows, w = patch cols
        #   → output: (batch, h*w_patches, embed_dim)
        x = rearrange(x, "b e h w -> b (h w) e")

        return x  # (B, 144, embed_dim)


# ─────────────────────────────────────────────────────────────────────────────
# BUILDING BLOCK 2: MULTI-HEAD SELF-ATTENTION
# ─────────────────────────────────────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    """
    The central mechanism of all Transformer models.

    INTUITION
    ─────────
    Every patch token asks: "Which other patches are relevant to me?"
    Each token creates a Query (what am I looking for?), a Key (what do I offer?),
    and a Value (what information do I carry?).

    The attention score between two tokens = Query_i · Key_j / √(head_dim).
    After softmax, this gives a probability distribution: "how much should
    patch i look at patch j?"
    The output is the weighted average of all Values.

    WHY MULTIPLE HEADS?
    ────────────────────
    Different heads can learn different notions of "relevance": one head might
    learn spatial proximity, another global scene structure, another might track
    the end-effector across frames.

    FORMULA (per head h)
    ─────────────────────
      Q_h = x W_Q_h,  K_h = x W_K_h,  V_h = x W_V_h
      A_h = softmax( Q_h K_h^T / √d_k )
      out_h = A_h V_h
      concat heads → project with W_O
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout:   float = 0.0,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads

        # Scaling factor prevents dot-products from growing too large,
        # which would push softmax into near-zero gradient territory.
        # Reference: "Attention Is All You Need", Vaswani et al. 2017
        self.scale = self.head_dim ** -0.5  # = 1 / √head_dim

        # Q, K, V projections fused into one matrix (3× the output dim)
        # Fusing is purely efficiency — the math is identical to three
        # separate linear layers.
        self.qkv  = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim,     bias=True)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x      : (B, N, embed_dim)   — N tokens (patches + CLS)
        returns: (B, N, embed_dim)
        """
        B, N, C = x.shape

        # Project to Q, K, V
        qkv = self.qkv(x)  # (B, N, 3 * embed_dim)

        # Rearrange: split last dim into (3, num_heads, head_dim) then unbind
        # Shape after rearrange: (3, B, num_heads, N, head_dim)
        qkv = rearrange(
            qkv,
            "b n (three h d) -> three b h n d",
            three=3, h=self.num_heads,
        )
        q, k, v = qkv.unbind(0)  # each: (B, num_heads, N, head_dim)

        # ── Scaled dot-product attention ──────────────────────────────────
        # (B, heads, N, d) @ (B, heads, d, N) → (B, heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # softmax over the KEY dimension (last dim)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Weighted sum of values: (B, heads, N, N) @ (B, heads, N, d)
        out = attn @ v  # (B, num_heads, N, head_dim)

        # Re-concat heads: (B, N, embed_dim)
        out = rearrange(out, "b h n d -> b n (h d)")

        # Final output projection
        out = self.proj(out)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# BUILDING BLOCK 3: MLP (Feed-Forward Network)
# ─────────────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """
    Two-layer feed-forward network applied to each token independently.

    INTUITION
    ─────────
    Attention mixes information across tokens.  The MLP then transforms each
    token's mixed representation in its own "thought space".  Together,
    attention + MLP = one Transformer block.

    The hidden dimension is mlp_ratio × embed_dim (typically 4×). This
    expansion-and-compression structure is common in LLMs and vision models.

    GELU vs ReLU
    ─────────────
    GELU (Gaussian Error Linear Unit) is smoother than ReLU near zero, which
    gives slightly better gradient flow.  Most modern transformers use GELU.
    """

    def __init__(
        self,
        embed_dim:  int,
        mlp_ratio:  float = 4.0,
        dropout:    float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),            # smooth non-linearity
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# BUILDING BLOCK 4: TRANSFORMER BLOCK (Attention + MLP with residuals)
# ─────────────────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    One full Transformer encoder layer.

    Structure (Pre-Norm variant, used in most modern models including ViT-22B)
    ─────────────────────────────────────────────────────────────────────────
      x = x + Attention( LayerNorm(x) )   ← residual + attention
      x = x + MLP(      LayerNorm(x) )   ← residual + MLP

    WHY RESIDUAL CONNECTIONS?
    ─────────────────────────
    Residual (skip) connections let gradients flow directly from the loss back
    to early layers without passing through every intermediate operation.
    This prevents the "vanishing gradient" problem and makes it possible to
    train deep networks (ResNet first proved this for CNNs; Transformers use it
    for the same reason).

    WHY LAYER NORM BEFORE (Pre-Norm) INSTEAD OF AFTER (Post-Norm)?
    ───────────────────────────────────────────────────────────────
    Pre-norm keeps activation scales stable throughout training, which means
    you can use higher learning rates and converge faster.  Almost all modern
    implementations (GPT, LLaMA, ViT-Large, etc.) use Pre-Norm.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout:   float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadSelfAttention(embed_dim, num_heads, dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp   = MLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual 1: attention branch
        x = x + self.attn(self.norm1(x))
        # Residual 2: MLP branch
        x = x + self.mlp(self.norm2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# BUILDING BLOCK 5: VISION ENCODER (the full ViT)
# ─────────────────────────────────────────────────────────────────────────────

class VisionEncoder(nn.Module):
    """
    Tiny Vision Transformer (ViT) — the full vision backbone of the VLA model.

    Output Contract
    ───────────────
    The model outputs two tensors:
      cls_token : (B, embed_dim)         — single global image representation
      patch_tokens : (B, num_patches, embed_dim) — per-patch spatial features

    BOTH are useful at different stages:
      - cls_token is used for contrastive alignment in pre-training
        (one vector per image → easy to compare with one vector per text)
      - patch_tokens are used for cross-attention fusion with language tokens
        in the fusion module (Baby Step 5), because they preserve WHERE
        in the image different features appear (spatial information)
    """

    def __init__(
        self,
        image_size:  int   = 84,
        patch_size:  int   = 7,
        in_channels: int   = 3,
        embed_dim:   int   = 128,
        num_heads:   int   = 4,
        num_layers:  int   = 4,
        mlp_ratio:   float = 4.0,
        dropout:     float = 0.0,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim

        # ── Component 1: Patch Embedding ──────────────────────────────────
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches   # 144 for 84×84, patch=7

        # ── Component 2: CLS token ────────────────────────────────────────
        # A learnable vector prepended to the patch sequence.
        # It has no spatial meaning — its job is to aggregate global context
        # from all patches via attention, producing a single image summary.
        # Introduced in the original ViT paper ("An Image is Worth 16×16 Words").
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # ── Component 3: Positional Embedding ─────────────────────────────
        # Transformers are permutation-invariant (they don't care about order).
        # We add learned position embeddings so the model knows WHERE each
        # patch is in the image.
        # Shape: (1, num_patches + 1, embed_dim)  — +1 for the CLS token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.pos_drop = nn.Dropout(dropout)

        # ── Component 4: Transformer Blocks ──────────────────────────────
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        # ── Component 5: Final LayerNorm ──────────────────────────────────
        self.norm = nn.LayerNorm(embed_dim)

        # ── Weight initialisation ─────────────────────────────────────────
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialise weights following the original ViT paper.

        Why not leave PyTorch's default (Kaiming uniform)?
        ────────────────────────────────────────────────────
        Kaiming is designed for ReLU networks.  Transformers use GELU and
        layer-norm, which have different variance dynamics.  The ViT paper
        uses truncated normal (std=0.02) for linear/conv weights and zeros
        for biases, which is also what GPT-2 and BERT use.
        """
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ──────────
        x : (B, C, H, W)  — batch of RGB images, pixel values in [0, 1]

        Returns
        ───────
        cls_token    : (B, embed_dim)              — global image summary
        patch_tokens : (B, num_patches, embed_dim) — spatial patch features
        """
        B = x.shape[0]

        # Step 1 — Patch embedding: (B, C, H, W) → (B, N, embed_dim)
        x = self.patch_embed(x)    # (B, 144, 128)

        # Step 2 — Prepend CLS token
        # expand(-1, -1, -1) keeps all dims the same except expands batch dim
        cls = self.cls_token.expand(B, -1, -1)   # (B, 1, embed_dim)
        x = torch.cat([cls, x], dim=1)            # (B, 145, embed_dim)

        # Step 3 — Add positional embedding (same shape as x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Step 4 — Transformer blocks
        for block in self.blocks:
            x = block(x)   # shape unchanged: (B, 145, embed_dim)

        # Step 5 — Final normalisation
        x = self.norm(x)   # (B, 145, embed_dim)

        # Step 6 — Split CLS token from patch tokens
        cls_token    = x[:, 0]       # (B, embed_dim)   — index 0 is CLS
        patch_tokens = x[:, 1:]      # (B, 144, embed_dim)

        return cls_token, patch_tokens

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# FACTORY FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def build_vision_encoder(
    config:     str   = "tiny",
    image_size: int   = 84,
    patch_size: int   = 7,
    dropout:    float = 0.0,
) -> VisionEncoder:
    """
    Convenience constructor.  Use this in training scripts so config changes
    are a one-word edit.

    Examples
    ────────
    >>> enc = build_vision_encoder("tiny")    # for quick experiments
    >>> enc = build_vision_encoder("small")   # better capacity
    """
    cfg = VIT_CONFIGS[config]
    return VisionEncoder(
        image_size  = image_size,
        patch_size  = patch_size,
        dropout     = dropout,
        **cfg,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST  (run this file directly to verify the encoder works)
# Usage:  python -m src.models.vision_encoder
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[vision_encoder test]  device = {device}")
    print("=" * 60)

    for config_name in VIT_CONFIGS:
        enc = build_vision_encoder(config_name).to(device)

        # Fake batch of 4 images: pixel values in [0, 1]
        imgs = torch.rand(4, 3, 84, 84, device=device)

        cls_out, patch_out = enc(imgs)

        n_params = enc.num_parameters()
        print(f"\n  Config: {config_name}")
        print(f"    Input shape         : {list(imgs.shape)}")
        print(f"    CLS output shape    : {list(cls_out.shape)}")
        print(f"    Patch output shape  : {list(patch_out.shape)}")
        print(f"    Trainable params    : {n_params:,}")
        print(f"    Memory (fwd pass)   : ~{imgs.element_size() * imgs.nelement() / 1e6:.2f} MB input")

        # Verify gradient flows end-to-end
        loss = cls_out.mean()
        loss.backward()
        print(f"    Backward pass       : OK")

    print("\n[vision_encoder test]  All configs passed.")
    print("\nKey shapes to understand:")
    print("  (B, embed_dim)            ← cls_token  — used for contrastive loss in pre-training")
    print("  (B, num_patches, embed_dim) ← patch_tokens — used for cross-attention in fusion")
    sys.exit(0)
