# Baby Step 10: Evaluation + Inference Demo

## Goal

Step 10 answers the only question that matters after training: does the robot actually behave better in rollout?

This stage compares checkpoints by running them in the same synthetic tasks and producing:
- success rate by task
- mean rollout MAE against oracle actions
- visual GIF demos
- a static HTML report for publishing

## What this repo generates

Running Step 10 writes:
- `outputs/eval/metrics.json`
- `outputs/eval/report.html`
- `outputs/eval/media/*.gif`

That report is meant to be easy to add to a GitHub repo, attach to releases, or host with GitHub Pages.

## Why this is better than only printing metrics

Numbers tell you whether training converged.
Visual rollouts tell you what the robot is actually trying to do.

For robotics-style projects, both are necessary.

## Checkpoint comparison

The default config compares:
- Step 8 SFT checkpoint
- Step 9 post-trained checkpoint

That makes it easy to show whether DAgger-lite actually improved robustness.

## Run

Full report:

```bash
python -m src.eval.evaluate --config configs/eval.yaml
```

Note: smoke configs were removed from the publishable repo to keep it clean. Use `configs/eval.yaml` for reproducible runs.

## Output style

The HTML report is intentionally presentation-friendly:
- summary metric cards
- per-task comparison table
- GIF gallery of rollout demos

That makes it useful both for debugging and for publishing results.