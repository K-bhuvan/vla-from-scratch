# Baby Step 7: Stage A Pre-Training - Contrastive Vision-Language Alignment

## Goal

Before teaching actions, we first teach the model to match the right image with the right instruction.

Given a batch of paired samples:
- positive pair: image_i with instruction_i
- negatives: image_i with instruction_j for all j != i

The model should score positive pairs higher than negatives.

## Why this stage exists

If the model cannot align language with visual content, behavior cloning in Step 8 has to learn both grounding and control at once. That is harder and less data efficient.

Stage A gives the model a grounded representation first, so Step 8 can focus on action prediction.

## Loss used

We use CLIP-style symmetric InfoNCE:

1. Encode image and instruction independently.
2. Project both to a shared embedding space.
3. L2-normalize embeddings.
4. Compute similarity matrix over the batch.
5. Apply cross-entropy in both directions.

This encourages each image to be closest to its matching instruction and vice versa.

## What is trained in Step 7

- VisionEncoder from Step 3
- LanguageEncoder from Step 4
- Two projection heads (vision and text)

Not trained here:
- Fusion module
- Action head

## Current repo note

The original Stage A implementation was used during development, but `src/train/pretrain.py` is not part of the current lightweight publishable repo path.

`configs/pretrain.yaml` is kept as a configuration reference.

## Run command

There is no Stage A runner in the current code path. Training flow starts from Step 8 (`src/train/sft.py`) in this published repo.

## Outputs

Checkpoints are saved to:

- outputs/pretrain/last.pt
- outputs/pretrain/best.pt

Both include model weights, optimizer state, current epoch, and config snapshot.
