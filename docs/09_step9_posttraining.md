# Baby Step 9: Stage C Post-Training - DAgger-lite

## Goal

Step 8 trains on expert trajectories only. Step 9 improves robustness by letting the current policy visit states, then asking the oracle what the correct action should have been there.

This reduces the train-test mismatch that happens when a cloned policy makes a small mistake and drifts into states the expert dataset rarely contains.

## What this implementation does

This repo uses a practical DAgger-lite variant:

1. Load the best Step 8 policy.
2. Roll it out on fresh synthetic scenes.
3. Query an oracle action at each visited state.
4. Keep correction samples where policy error is above a threshold.
5. Fine-tune the policy on those correction samples.

## Why it is called "lite"

Classic DAgger aggregates a growing dataset across multiple rollout-train cycles and may mix full expert and correction distributions.

This MVP keeps one focused correction-collection pass followed by one fine-tuning pass. That is enough to teach the core idea without building a full online robotics infrastructure.

## Inputs and targets

Input at each visited policy state:
- rendered image from current simulator state
- instruction text

Target:
- oracle action for that visited state

## Outputs

- outputs/posttrain/corrections.pt
- outputs/posttrain/best.pt
- outputs/posttrain/last.pt

## Run

Full run:

```bash
python -m src.posttrain.dagger --config configs/posttrain.yaml
```

Note: smoke configs were removed from the publishable repo to keep it clean. Use `configs/posttrain.yaml` for reproducible runs.

## Main difference from Step 8

- Step 8 learns from static expert data.
- Step 9 learns from policy-induced states and oracle corrections.

That is the key reason Step 9 improves recovery behavior and rollout robustness.