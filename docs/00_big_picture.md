# The Big Picture: How This Repo Mirrors an Industry VLA Pipeline

> Read this before touching any code.
> It will help you understand *why* each component exists.

## Step Docs

Use this file for the overall roadmap, then read the step-specific notes after
each implementation step:

- Step 1: `docs/01_step1_setup.md`
- Step 2: `docs/02_step2_data_generation.md`
- Step 3: `docs/03_step3_vision_encoder.md`

From this point onward, every new implementation step will also get its own
separate markdown explainer in this folder.

---

## What large robotics companies actually do

Companies like Google DeepMind, Physical Intelligence, Skild AI, and Apptronik
train VLA models roughly like this:

```
Massive robot fleet
  collects millions of demonstrations
         │
         ▼
Foundation model training
  (vision + language alignment, broad generalization)
         │
         ▼
Policy fine-tuning
  (specific tasks, specific robots, instruction following)
         │
         ▼
Online improvement (data flywheel)
  (deploy policy → collect failures → re-label → retrain → repeat)
         │
         ▼
Safety evaluation + deployment
  (benchmark suites, red-teaming, latency requirements, model card)
```

This repo replicates the same **logic** but swaps:
- Robot fleet → open-source simulation (ManiSkill)
- Millions of demos → tens of thousands of simulation demos
- Proprietary model → 80M–120M parameter model you build from scratch
- Cloud cluster → your single 16GB GPU

---

## Stage A: Pre-Training (Representation + Grounding)

**Industry analogy:** Before a VLA can act, it must *understand* the world.
Large labs often start with a powerful vision-language model (like an LLM + vision encoder)
and then connect it to robot actions. They pre-train on internet-scale image-text pairs
first, so the model already "speaks the same language" as task instructions.

**What we do:** We train a small vision-language backbone from scratch using contrastive
learning. Given an image of a scene and its task description, the model learns to
produce similar embedding vectors for matching pairs and dissimilar ones for non-matching pairs.

```
Image: robot arm near a cup               ──► embedding_img
Text:  "pick up the red cup"              ──► embedding_text

Loss: push embedding_img and embedding_text CLOSER TOGETHER
      push them away from unrelated (image, text) pairs
```

This is the same core idea as CLIP (by OpenAI).

**What you learn:** positional encodings, patch embeddings, contrastive loss, cosine similarity.

---

## Stage B: Supervised Fine-Tuning (SFT)

**Industry analogy:** Once the backbone understands the world, teams fine-tune it on
robot demonstrations. Expert operators (or scripted oracles) collect trajectories:
at each timestep, they record the camera image, the instruction, and the action taken.
The model is trained to predict those expert actions — this is called **behavior cloning (BC)**.

**What we do:** We feed the pre-trained backbone (frozen or lightly thawed) our simulated
demonstration dataset. The training signal is:
```
predicted_action = policy(image, instruction, action_history)
loss = mean_squared_error(predicted_action, expert_action)
```

**What you learn:** teacher forcing, sequence modeling, BC limitations (compounding errors).

---

## Stage C: Post-Training (Policy Improvement Loop)

**Industry analogy:** BC alone produces a brittle policy — it only saw perfect
expert states during training, so it does not know how to recover from mistakes.
Large labs run **online improvement loops**: deploy the current policy, let it fail,
collect human or oracle corrections on those failure states, add them to training data,
retrain. This is the "data flywheel."

**What we do:** A simplified version called **DAgger-lite**:
1. Roll out the current policy in simulation
2. Record states where it fails
3. Query a scripted oracle for the correct action at those states
4. Add those (state, oracle_action) pairs to the training set
5. Fine-tune the policy on the expanded dataset
6. Repeat 1–5 cycles

**What you learn:** distribution shift, why BC fails in practice, online data collection.

---

## Stage D: Evaluation and Release

**Industry analogy:** Before shipping, companies run structured benchmarks across
diverse task variants, measure success rates, document failure modes, and publish
a model card. They also enforce latency and memory budgets for real-time deployment.

**What we do:**
- Fixed-seed benchmark episodes across N tasks
- Log success rate, action MSE, inference latency
- Write a model card (what the model can/can't do, training data, known biases)
- A demo inference script for GitHub

**What you learn:** reproducible evaluation setup, model cards, failure analysis.

---

## The Neural Architecture (Preview)

```
┌─────────────────────────────────────────────────────────┐
│                     VLA Model (MVP)                     │
│                                                         │
│  Image ──► [Vision Encoder]  ──► image tokens           │
│                                        │                │
│  Text  ──► [Language Encoder] ──► text tokens           │
│                                        │                │
│                     ┌──────────────────┘                │
│                     ▼                                   │
│               [Fusion Module]   (cross-attention)       │
│                     │                                   │
│                     ▼                                   │
│               [Action Head]     (MLP or transformer)    │
│                     │                                   │
│                     ▼                                   │
│              predicted action                           │
└─────────────────────────────────────────────────────────┘
```

Each box is one Baby Step in the implementation plan.

---

## Why build from scratch instead of fine-tuning a pretrained model?

For learning, building from scratch is more valuable because:
1. You see exactly how each component works — nothing is hidden.
2. You make every design decision consciously instead of inheriting them.
3. You can trace any training bug to its source.
4. In interviews, you can explain every layer you wrote.

The trade-off: lower task performance. That is acceptable for an MVP.
