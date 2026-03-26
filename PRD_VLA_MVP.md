# Product Requirements Document (PRD)
## Project: Local MVP Vision-Language-Action (VLA) Training From Scratch
## Date: 2026-03-22

## 1) Purpose
Build a learning-first, resume-ready MVP repository that demonstrates the end-to-end VLA lifecycle on a single local GPU (NVIDIA RTX-class 16GB VRAM), including:
- pre-training (representation and multimodal grounding)
- supervised fine-tuning (SFT) for instruction-to-action behavior
- post-training (robustness and policy improvement loop)
- evaluation, safety checks, and deployment-style inference demo

The repository should prioritize understanding and reproducibility over leaderboard performance.

## 2) Problem Statement
Large robotics AI teams train VLA systems with massive compute and proprietary robot fleets. This project replicates the same training logic at small scale:
- Stage A: learn visual and language grounding
- Stage B: learn action generation from demonstrations
- Stage C: improve policy robustness after initial training
- Stage D: package and evaluate as if preparing for real users

The outcome is not a production robot foundation model. It is a pedagogical VLA pipeline that mirrors the industry process in miniature.

## 3) Target User
Primary user: You (and recruiters/interviewers reviewing your GitHub repo).
Secondary user: engineers who want a practical VLA learning template on consumer hardware.

## 4) Goals
- Learn full VLA lifecycle from raw data to final policy checkpoints.
- Run each stage on a single 16GB GPU.
- Keep code modular and easy to read.
- Use only open-source, free-to-use tooling and data sources.
- Produce measurable outputs and experiment logs for each stage.

## 5) Non-Goals
- State-of-the-art robotics manipulation performance.
- Real-world hardware deployment in this MVP phase.
- Multi-GPU distributed training.
- Proprietary datasets or closed APIs.

## 6) Hardware and Constraints
Assumed machine:
- 1x NVIDIA RTX GPU, 16GB VRAM
- 32GB RAM
- 200GB+ free disk for datasets/checkpoints/logs
- Windows host acceptable, but Linux/WSL training environment preferred

Key constraints:
- Full end-to-end from-scratch training of large (>500M) VLA is not practical.
- Training must use mixed precision, gradient accumulation, and efficient dataloading.

## 7) Recommended Model Sizes For This Machine
Practical parameter targets for learning-focused training:

1. Tiny VLA (best for first complete run)
- total params: 30M to 80M
- expected behavior: stable training, fast iteration, lower quality actions

2. Small VLA (best showcase target)
- total params: 80M to 180M
- expected behavior: meaningful instruction conditioning, still trainable on 16GB with careful batch/sequence settings

3. Upper bound experiment (optional)
- total params: 180M to 300M
- feasible only with aggressive memory optimization (activation checkpointing, low micro-batch)
- significantly slower iteration and higher OOM risk

Recommendation for MVP:
- Start with 80M to 120M total parameters.
- Keep action head compact.
- Prefer short sequence horizons first, then extend.

## 8) System Overview (Industry-Style, Scaled Down)
High-level flow:
1. Data engine setup
2. Pre-training for multimodal representations
3. SFT for instruction-conditioned action prediction
4. Post-training with robustness loop (DAgger-like and/or preference reranking)
5. Evaluation and safety gating
6. Inference demo and reproducible report

Pipeline analogy to large companies:
- Foundation stage: broad multimodal representation learning
- Policy stage: behavior cloning / instruction tuning
- Improvement stage: iterative data flywheel (collect failures -> relabel -> retrain)
- Productization stage: monitor metrics, enforce safety checks, release a stable policy

## 9) Data Strategy (Open and Free)
Important legal principle for GitHub:
- do not re-upload raw datasets
- provide download scripts + LICENSE/NOTICE references
- keep a dataset provenance table in repo

Preferred MVP data strategy (safest):
1. Self-generated simulation trajectories (primary)
- simulator: ManiSkill and/or robosuite-like open simulation stack
- advantage: fully reproducible, no redistribution issues if scripts only

2. Public robot imitation datasets (secondary)
- use only subsets with clear open terms
- include per-dataset citation and usage terms in docs

3. Language supervision
- template-generated task instructions from simulator metadata
- optional paraphrase augmentation using open models

Candidate open/free sources to evaluate in implementation phase:
- ManiSkill generated demos (simulation-generated)
- RLBench-style generated demonstrations (if license-compatible in your setup)
- RoboMimic compatible demo formats (use open subsets only)
- BridgeData/Open-X style subsets only after explicit license check

## 10) Training Stages and Deliverables
### Stage A: Pre-Training (Representation + Grounding)
Objective:
- learn joint vision-language features and temporal context before strong action supervision

Methods (MVP-friendly):
- contrastive alignment: image/clip <-> task text
- temporal consistency: adjacent observation embedding similarity
- masked prediction auxiliary losses (optional)

Output:
- pretrained vision-language backbone checkpoint
- representation quality metrics (retrieval accuracy, contrastive loss curves)

### Stage B: SFT (Instruction -> Action)
Objective:
- train policy to map (observation, instruction, history) -> action tokens/continuous action

Methods:
- behavior cloning loss (MSE for continuous controls or cross-entropy for action tokens)
- teacher forcing over trajectory chunks
- balanced sampling across tasks

Output:
- SFT policy checkpoint
- task success on held-out simulation episodes

### Stage C: Post-Training (Policy Improvement)
Objective:
- improve robustness and reduce compounding errors

Methods (pick one or both):
1. DAgger-lite loop
- roll out policy in simulator
- collect failure states
- relabel using scripted oracle or stronger teacher policy
- continue fine-tuning

2. Preference/reranking objective (lightweight)
- generate multiple candidate action chunks
- score via success heuristics
- optimize to prefer better trajectories

Output:
- post-trained checkpoint
- improved robustness metrics vs SFT baseline

### Stage D: Evaluation and Release
Objective:
- package results as a product-like release candidate

Methods:
- fixed benchmark tasks + random seeds
- latency and throughput logging
- failure case taxonomy
- model card and known limitations

Output:
- evaluation report
- inference demo script
- model card

## 11) Metrics (What Success Looks Like)
Core metrics:
- task success rate (% episodes completed)
- action prediction loss (validation)
- robustness under perturbations (camera jitter, initial state variation)
- inference latency (ms/step)
- training stability (loss spikes, divergence rate)

MVP success criteria:
- complete pipeline runs end-to-end without manual patching
- post-training checkpoint improves success rate over SFT by >= 5 absolute points on selected tasks
- reproducible run with documented seeds/configs
- clean repository narrative suitable for interview walkthrough

## 12) Estimated Training Time and GPU Utilization (Single 16GB GPU)
These are planning estimates for an 80M to 120M parameter VLA with simulation data.
Actual runtime depends heavily on dataloader speed, sequence length, and simulator throughput.

1. Stage A pre-training
- data scale: 200k to 600k multimodal samples
- time: ~8 to 24 hours
- expected GPU utilization: 70% to 95% (high when data pipeline is efficient)

2. Stage B SFT
- data scale: 100k to 300k trajectory chunks
- time: ~6 to 16 hours
- expected GPU utilization: 75% to 95%

3. Stage C post-training
- rollout + relabel + fine-tune cycles: 2 to 6 cycles
- total time: ~8 to 30 hours
- expected GPU utilization: 40% to 85%
- note: rollout/relabel phases can be CPU/simulator bottleneck, lowering average GPU usage

4. Full MVP pipeline end-to-end
- total elapsed project runtime for one serious run: ~1.5 to 4 days
- engineering iteration timeline (including debugging/analysis): ~2 to 4 weeks part-time

## 13) Memory/Batching Guidance for 16GB
Recommended defaults:
- mixed precision: bf16 (or fp16 if needed)
- gradient accumulation: enabled
- activation checkpointing: enabled for >100M models
- micro-batch size: 1 to 8 (task dependent)
- sequence/chunk length: start small (8 to 16 frames), expand later

OOM mitigation order:
1. reduce sequence length
2. reduce micro-batch size
3. enable/expand checkpointing
4. reduce hidden dimension / number of layers

## 14) Repository Structure (Learning-First)
Planned structure:
- docs/: architecture notes, dataset cards, experiment reports
- configs/: per-stage yaml configs
- data/: download scripts + dataset metadata only
- src/models/: encoders, fusion module, action head
- src/train/: stage-specific trainers
- src/eval/: benchmarking and rollout eval
- src/posttrain/: dagger/preference loops
- scripts/: one-command stage launchers
- notebooks/: analysis and visualization

Learning-first requirements:
- each stage has an explainer document
- each trainer logs key tensors/metrics
- one-page "How this mirrors industry pipeline" write-up

## 15) Risks and Mitigations
Risk: simulator throughput bottlenecks
- mitigation: pre-generate trajectory datasets for early runs

Risk: unstable training
- mitigation: start with tiny model and overfit-on-small-subset test first

Risk: unclear legal status of some datasets
- mitigation: default to self-generated simulation data and explicit license table

Risk: over-scoping
- mitigation: strict milestone gates and stage completion checklist

## 16) Milestones
M0 (Setup + sanity)
- environment setup, tiny overfit test, logging baseline

M1 (Pre-training complete)
- backbone checkpoint + representation report

M2 (SFT complete)
- instruction-conditioned policy + baseline success rate

M3 (Post-training complete)
- robustness-improved policy + comparison report

M4 (Release-ready repo)
- docs, reproducibility script, model card, demo video/gif plan

## 17) Acceptance Criteria for PRD Sign-Off
- You agree with scope and learning-first focus.
- You approve initial target model size (80M to 120M).
- You approve simulation-first dataset strategy.
- You approve stage sequence: pre-train -> SFT -> post-train -> release eval.
- You approve estimated runtime envelope and MVP milestones.

## 18) What Happens After Approval
Implementation plan (next step after your confirmation):
1. convert PRD into technical design doc with exact model architecture
2. finalize dataset list and licenses in a provenance table
3. scaffold repository and training config system
4. implement Stage A trainer first, then Stage B/C
5. run baseline experiments and produce first report
