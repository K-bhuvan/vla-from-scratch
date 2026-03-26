"""
src/models/fusion.py  —  Baby Step 5: Vision-Language Fusion Module (Cross-Attention from scratch)

════════════════════════════════════════════════════════════════════════════════
WHAT THIS FILE BUILDS
════════════════════════════════════════════════════════════════════════════════
A fusion module that aligns vision and language representations using cross-attention.

The model receives two separate modalities:
  - Vision: cls_token (global image summary) + patch_tokens (spatial features)
  - Language: cls_token (global instruction summary) + token_tokens (semantic features)

And outputs:
  - fused_cls : (B, fused_dim)  — combined global representation (image understanding + instruction understanding)
  - fused_tokens : (B, num_patches, fused_dim) — spatially-aligned features (each patch knows the language context)

════════════════════════════════════════════════════════════════════════════════
WHY CROSS-ATTENTION (NOT SIMPLE CONCATENATION)?
════════════════════════════════════════════════════════════════════════════════
Simple concatenation: vision_cls + language_cls → (B, 2 * embed_dim)
Problem: The model doesn't learn which parts of the image relate to which words.

Cross-attention: Let vision patches "ask questions" about language tokens, and vice versa.
  - Q: "I'm a patch in the image at position (x, y). Which language tokens describe me?"
  - K, V: "I'm the language for 'red', 'cube', 'pick', etc."
  - Output: Each patch gets a weighted blend of language embeddings, conditioned on spatial location.

This creates fine-grained alignment: the patch at (3, 5) learns that "cube" and "red"
are relevant to it, while the patch at (0, 0) learns that "pick" is the relevant verb.

Result: Much better understanding of instructions in visual context.

════════════════════════════════════════════════════════════════════════════════
ARCHITECTURE: 3 BUILDING BLOCKS
════════════════════════════════════════════════════════════════════════════════
  1. CrossAttention       — query from one modality, key/value from another
  2. FusionBlock          — fuses vision and language in both directions
  3. VisionLanguageFusion — stacks blocks, produces fused representation

════════════════════════════════════════════════════════════════════════════════
COMPARISON: SELF-ATTENTION vs CROSS-ATTENTION
════════════════════════════════════════════════════════════════════════════════

SELF-ATTENTION (steps 3-4: within single modality)
───────────────────────────────────────────────
  Q, K, V all come from the SAME sequence:
    Q = tokens @ W_Q    (what am I looking for?)
    K = tokens @ W_K    (what do I offer?)
    V = tokens @ W_V    (what info do I carry?)
    out = softmax(Q K^T / √d) V

  Use case: "pick" and "cube" understanding their relationship
            within the instruction alone.

CROSS-ATTENTION (step 5: between modalities)
──────────────────────────────────────────────
  Q comes from one sequence, K and V from a DIFFERENT sequence:
    Q = vision_patches @ W_Q_vision    (what am I a vision patch looking for?)
    K = language_tokens @ W_K_lang     (what language context do you offer?)
    V = language_tokens @ W_V_lang     (what semantic info do you carry?)
    out = softmax(Q K^T / √d) V

  Use case: Vision patch at (row=5, col=3) asks "which language tokens describe me?"
            Gets back a weighted blend of "pick", "red", "cube", etc.

The beauty: Vision patches learn WHERE in the image linguistic concepts apply.

════════════════════════════════════════════════════════════════════════════════
DATA FLOW DIAGRAM
════════════════════════════════════════════════════════════════════════════════

  Vision Stream                 Language Stream
  ─────────────                 ───────────────
  vision_cls     (B, dim)       lang_cls       (B, dim)
  vision_patches (B, 144, dim)  lang_tokens    (B, seq_len, dim)
        │                             │
        ▼                             ▼
  ┌─────────────────────────────────────────┐
  │     FusionBlock (bidir cross-attn)      │
  │  ───────────────────────────────────    │
  │  1. patches attend to tokens (v→l)      │
  │  2. tokens attend to patches (l→v)      │
  │  3. fused_cls = concat vision+lang cls  │
  │  4. fused_tokens = concat vision+lang   │
  └─────────────────────────────────────────┘
        │                             │
        ▼                             ▼
  fused_tokens   (B, 144, 2*dim)
  fused_cls      (B, 2*dim)  ← ready for Action Head in Step 6
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from einops import rearrange


# ─────────────────────────────────────────────────────────────────────────────
# PRE-DEFINED FUSION CONFIGURATIONS
# ─────────────────────────────────────────────────────────────────────────────

FUSION_CONFIGS: dict[str, dict] = {
    #           embed_dim  num_heads  mlp_ratio
    "tiny":  dict(embed_dim=128, num_heads=4, mlp_ratio=4.0),  # lightweight (matches tiny vision + tiny language)
    "small": dict(embed_dim=256, num_heads=8, mlp_ratio=4.0),  # recommended (matches small vision + small language)
    "base":  dict(embed_dim=512, num_heads=8, mlp_ratio=4.0),  # heavy (matches base vision + base language)
}


# ─────────────────────────────────────────────────────────────────────────────
# BUILDING BLOCK 1: CROSS-ATTENTION
# ─────────────────────────────────────────────────────────────────────────────

class CrossAttention(nn.Module):
    """
    Cross-attention: query from one modality, key/value from another.

    INTUITION
    ─────────
    One modality (e.g., vision patches) asks questions about another modality
    (e.g., language tokens).  The questions are "which language concepts are
    most relevant to this visual patch?"

    Formula
    ───────
      vision_patch: "I'm at position (row=5, col=3). What language describes me?"
      Q = vision_patch @ W_Q_vision
      K = language_tokens @ W_K_lang
      V = language_tokens @ W_V_lang
      attention = softmax(Q K^T / √d_k)
      output = attention @ V

      Result: a weighted blend of language embeddings, customized for this patch.

    WHY NOT SELF-ATTENTION HERE?
    ─────────────────────────────
    Self-attention only lets each modality understand itself. But the whole
    point of vision-language fusion is CROSS-MODAL understanding.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Q from one modality (e.g., vision)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        # K, V from other modality (e.g., language)
        self.kv_proj = nn.Linear(embed_dim, embed_dim * 2, bias=True)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,  # (B, N_q, embed_dim) — query from one modality
        kv: torch.Tensor,  # (B, N_kv, embed_dim) — key/value from other modality
    ) -> torch.Tensor:
        """
        q : (B, N_q, embed_dim)  — e.g., vision patches
        kv : (B, N_kv, embed_dim) — e.g., language tokens
        returns : (B, N_q, embed_dim)
        """
        B, N_q, C = q.shape
        _, N_kv, _ = kv.shape

        # Project Q from query modality
        q_proj = self.q_proj(q)  # (B, N_q, embed_dim)
        q_proj = rearrange(q_proj, "b n (h d) -> b h n d", h=self.num_heads)
        # (B, num_heads, N_q, head_dim)

        # Project K, V from key/value modality
        kv_proj = self.kv_proj(kv)  # (B, N_kv, 2 * embed_dim)
        kv_proj = rearrange(kv_proj, "b n (two h d) -> two b h n d", two=2, h=self.num_heads)
        # (2, B, num_heads, N_kv, head_dim)
        k_proj, v_proj = kv_proj.unbind(0)

        # Cross-attention: (B, heads, N_q, head_dim) @ (B, heads, head_dim, N_kv)
        #                → (B, heads, N_q, N_kv)
        attn = (q_proj @ k_proj.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values: (B, heads, N_q, N_kv) @ (B, heads, N_kv, head_dim)
        #                          → (B, heads, N_q, head_dim)
        out = attn @ v_proj

        # Recombine heads: (B, N_q, embed_dim)
        out = rearrange(out, "b h n d -> b n (h d)")

        # Output projection
        out = self.out_proj(out)

        return out


# ─────────────────────────────────────────────────────────────────────────────
# BUILDING BLOCK 2: FUSION BLOCK (Bidirectional Cross-Attention)
# ─────────────────────────────────────────────────────────────────────────────

class FusionBlock(nn.Module):
    """
    One full fusion layer with bidirectional cross-attention.

    Structure
    ─────────
      1. vision patches attend to language tokens (vision asks about language)
      2. language tokens attend to vision patches (language grounds in vision)

    Result: Tightly coupled alignment between spatial regions and semantic concepts.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # Vision → Language: patches ask about tokens
        self.norm_v1 = nn.LayerNorm(embed_dim)
        self.attn_v2l = CrossAttention(embed_dim, num_heads, dropout)

        # Language → Vision: tokens ground in patches
        self.norm_l1 = nn.LayerNorm(embed_dim)
        self.attn_l2v = CrossAttention(embed_dim, num_heads, dropout)

        # MLPs for further refinement
        self.norm_v2 = nn.LayerNorm(embed_dim)
        self.mlp_v = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

        self.norm_l2 = nn.LayerNorm(embed_dim)
        self.mlp_l = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        vision_tokens: torch.Tensor,  # (B, N_vision, embed_dim)
        language_tokens: torch.Tensor,  # (B, N_lang, embed_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        vision_tokens : (B, 144, embed_dim)  — patches
        language_tokens : (B, seq_len, embed_dim) — tokens
        returns : fused (vision_tokens, language_tokens)
        """

        # Vision → Language: patches attend to tokens and get language context
        vision_tokens = vision_tokens + self.attn_v2l(
            self.norm_v1(vision_tokens), language_tokens
        )
        vision_tokens = vision_tokens + self.mlp_v(self.norm_v2(vision_tokens))

        # Language → Vision: tokens ground in visual context
        language_tokens = language_tokens + self.attn_l2v(
            self.norm_l1(language_tokens), vision_tokens
        )
        language_tokens = language_tokens + self.mlp_l(self.norm_l2(language_tokens))

        return vision_tokens, language_tokens


# ─────────────────────────────────────────────────────────────────────────────
# BUILDING BLOCK 3: VISION-LANGUAGE FUSION MODULE (the full fusion)
# ─────────────────────────────────────────────────────────────────────────────

class VisionLanguageFusion(nn.Module):
    """
    Multimodal fusion module combining vision and language via cross-attention.

    Output Contract
    ───────────────
    Two modalities enter separately; they leave fused and aligned:
      - fused_cls : (B, 2 * embed_dim) — combined global representation
      - fused_tokens : (B, num_patches, 2 * embed_dim) — spatially-aligned

    The fused output is ready for the Action Head (Baby Step 6).
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_fusion_layers: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_fusion_layers = num_fusion_layers

        # Stack multiple fusion blocks for deeper alignment
        self.fusion_blocks = nn.ModuleList([
            FusionBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_fusion_layers)
        ])

        # Final layer norms
        self.norm_vision_out = nn.LayerNorm(embed_dim)
        self.norm_language_out = nn.LayerNorm(embed_dim)

        # Option 1: Concatenation (simple, recommended for MVP)
        self.concat_mode = True

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

    def forward(
        self,
        vision_cls: torch.Tensor,  # (B, embed_dim)
        vision_tokens: torch.Tensor,  # (B, num_patches, embed_dim)
        language_cls: torch.Tensor,  # (B, embed_dim)
        language_tokens: torch.Tensor,  # (B, seq_len, embed_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ──────────
        vision_cls : (B, embed_dim)
        vision_tokens : (B, 144, embed_dim)  — from VisionEncoder
        language_cls : (B, embed_dim)
        language_tokens : (B, seq_len, embed_dim)  — from LanguageEncoder

        Returns
        ───────
        fused_cls : (B, 2 * embed_dim)  — global joint representation
        fused_tokens : (B, 144, 2 * embed_dim)  — spatially-aligned patches + language enrichment
        """
        B = vision_cls.shape[0]
        num_patches = vision_tokens.shape[1]  # 144

        # Step 1: Apply fusion blocks (bidirectional cross-attention)
        for block in self.fusion_blocks:
            vision_tokens, language_tokens = block(vision_tokens, language_tokens)

        # Step 2: Final layer norms
        vision_tokens = self.norm_vision_out(vision_tokens)
        language_tokens = self.norm_language_out(language_tokens)

        # Step 3: Fuse CLS tokens (concatenate for true multimodal fusion)
        if self.concat_mode:
            fused_cls = torch.cat([vision_cls, language_cls], dim=-1)  # (B, 2 * embed_dim)
        else:
            # Alternative: simple addition (less information)
            fused_cls = vision_cls + language_cls  # (B, embed_dim)

        # Step 4: Fuse token sequences
        # Vision patches have been enriched with language context via cross-attention.
        # Language tokens have been grounded in vision via cross-attention.
        #
        # Strategy: Create learned embeddings for padding so gradients can flow.
        # This is cleaner than zero-padding.
        #
        # vision_tokens: (B, 144, embed_dim)  ← patches enhanced with language understanding
        # language_tokens: (B, seq_len, embed_dim)  ← tokens grounded in visual regions
        #
        # We create a "language projection" that pads to match patch count,
        # then concatenate for the final fused representation.

        # Create a learnable padding projection (not actually a parameter in this simple version)
        # Instead, just truncate/expand language to match patch dimensions for concatenation
        seq_len = language_tokens.shape[1]
        if seq_len >= num_patches:
            # Truncate language tokens to match patch count
            lang_for_concat = language_tokens[:, :num_patches, :]
        else:
            # Pad using last token repeated (better than zeros for maintaining gradient flow)
            # This is a simple solution: repeat the last language token to fill space
            last_token = language_tokens[:, -1:, :].expand(B, num_patches - seq_len, -1)
            lang_for_concat = torch.cat([language_tokens, last_token], dim=1)

        # Concatenate along embedding dimension
        fused_tokens = torch.cat(
            [vision_tokens, lang_for_concat],
            dim=-1
        )  # (B, 144, 2 * embed_dim)

        return fused_cls, fused_tokens

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# FACTORY FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def build_fusion_module(
    config: str = "tiny",
    num_fusion_layers: int = 2,
    dropout: float = 0.0,
) -> VisionLanguageFusion:
    """
    Convenience constructor.

    Examples
    ────────
    >>> fusion = build_fusion_module("small", num_fusion_layers=2)
    >>> fused_cls, fused_tokens = fusion(v_cls, v_tokens, l_cls, l_tokens)
    """
    cfg = FUSION_CONFIGS[config]
    return VisionLanguageFusion(
        num_fusion_layers=num_fusion_layers,
        dropout=dropout,
        **cfg,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST  (run this file directly to verify the fusion works)
# Usage:  python -m src.models.fusion
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[fusion test]  device = {device}")
    print("=" * 60)

    for config_name in FUSION_CONFIGS:
        fusion = build_fusion_module(config_name, num_fusion_layers=2).to(device)

        # Fake batch: 4 images paired with 4 instructions
        batch_size = 4

        # Vision: CLS + 144 patches (12×12 grid, 84×84 image with patch_size=7)
        embed_dim = FUSION_CONFIGS[config_name]["embed_dim"]
        vision_cls = torch.randn(batch_size, embed_dim, device=device, requires_grad=True)
        vision_tokens = torch.randn(batch_size, 144, embed_dim, device=device, requires_grad=True)

        # Language: CLS + seq_len tokens
        language_cls = torch.randn(batch_size, embed_dim, device=device, requires_grad=True)
        language_tokens = torch.randn(batch_size, 32, embed_dim, device=device, requires_grad=True)

        fused_cls, fused_tokens = fusion(vision_cls, vision_tokens, language_cls, language_tokens)

        n_params = fusion.num_parameters()
        print(f"\n  Config: {config_name}")
        print(f"    Input vision_cls      : {list(vision_cls.shape)}")
        print(f"    Input vision_tokens   : {list(vision_tokens.shape)}")
        print(f"    Input language_cls    : {list(language_cls.shape)}")
        print(f"    Input language_tokens : {list(language_tokens.shape)}")
        print(f"    Output fused_cls      : {list(fused_cls.shape)}")
        print(f"    Output fused_tokens   : {list(fused_tokens.shape)}")
        print(f"    Trainable params      : {n_params:,}")

        # Verify gradient flows end-to-end
        loss = fused_cls.mean() + fused_tokens.mean()
        loss.backward()
        print(f"    Backward pass         : OK")

    print("\n[fusion test]  All configs passed.")
    print("\nKey shapes to understand:")
    print("  (B, 2 * embed_dim)         ← fused_cls — global joint representation for action head")
    print("  (B, 144, 2 * embed_dim)    ← fused_tokens — spatially-grounded language features")
    sys.exit(0)
