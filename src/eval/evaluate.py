"""
src/eval/evaluate.py  -  Baby Step 10: evaluation + visual inference demo

Generates:
- rollout metrics by task
- checkpoint comparison (Step 8 vs Step 9)
- GIF demos for each task
- a static HTML report that is easy to publish in the repo
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from data.generate_sim_data import TASKS
from src.posttrain.dagger import (
    apply_policy_action,
    build_rollout_scene,
    is_success,
    oracle_action_for_state,
    rollout_frame,
    set_seed,
    tokenize_instruction,
    update_goal_delta,
)
from src.train.sft import VLABehaviorCloningModel


def load_config(config_path: str | Path) -> dict[str, Any]:
    cfg = OmegaConf.load(config_path)
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def load_policy(model_cfg: dict[str, Any], checkpoint_path: Path, device: torch.device) -> VLABehaviorCloningModel:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = VLABehaviorCloningModel(
        model_size=str(model_cfg["size"]),
        num_fusion_layers=int(model_cfg["num_fusion_layers"]),
        dropout=float(model_cfg["dropout"]),
        action_dim=int(model_cfg["action_dim"]),
        use_state=bool(model_cfg.get("use_state", False)),
        state_dim=int(model_cfg.get("state_dim", 7)),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def save_gif(frames: list[np.ndarray], path: Path, upscale: int, duration_ms: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames = [Image.fromarray(frame).resize((frame.shape[1] * upscale, frame.shape[0] * upscale), Image.Resampling.NEAREST) for frame in frames]
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )


def run_rollout(
    model: VLABehaviorCloningModel,
    task: str,
    rng: np.random.Generator,
    device: torch.device,
    eval_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    save_demo: bool,
    gif_path: Path | None,
) -> dict[str, Any]:
    scene = build_rollout_scene(task, rng)
    vocab_size = int(model_cfg["vocab_size"])
    max_seq_len = int(model_cfg["max_seq_len"])
    max_steps = int(eval_cfg["max_steps"])
    action_clip = float(eval_cfg["action_clip"])
    action_clip_gripper = float(eval_cfg.get("action_clip_gripper", action_clip))
    frames: list[np.ndarray] = []
    rollout_maes: list[float] = []

    for _ in range(max_steps):
        frame = rollout_frame(scene)
        if save_demo:
            frames.append(frame)

        image = torch.from_numpy(frame).permute(2, 0, 1).float().div(255.0)
        token_ids = tokenize_instruction(scene.instruction, vocab_size, max_seq_len)

        with torch.no_grad():
            pred_action = model(
                image.unsqueeze(0).to(device),
                token_ids.unsqueeze(0).to(device),
                torch.from_numpy(scene.state.copy()).unsqueeze(0).to(device),
            )[0].detach().cpu()

        oracle_action = torch.from_numpy(oracle_action_for_state(scene))
        rollout_maes.append(float(torch.mean(torch.abs(pred_action - oracle_action)).item()))
        apply_policy_action(
            scene,
            pred_action.numpy(),
            action_clip,
            action_clip_gripper=action_clip_gripper,
        )
        update_goal_delta(scene)

        if is_success(scene):
            final_frame = rollout_frame(scene)
            if save_demo:
                frames.append(final_frame)
            if save_demo and gif_path is not None:
                save_gif(frames, gif_path, upscale=int(eval_cfg["gif_upscale"]), duration_ms=int(eval_cfg["gif_duration_ms"]))
            return {
                "success": True,
                "steps": len(rollout_maes),
                "mean_rollout_mae": float(np.mean(rollout_maes)),
                "instruction": scene.instruction,
                "gif": gif_path.name if gif_path is not None else None,
            }

    if save_demo and gif_path is not None:
        save_gif(frames, gif_path, upscale=int(eval_cfg["gif_upscale"]), duration_ms=int(eval_cfg["gif_duration_ms"]))

    return {
        "success": False,
        "steps": len(rollout_maes),
        "mean_rollout_mae": float(np.mean(rollout_maes)) if rollout_maes else float("nan"),
        "instruction": scene.instruction,
        "gif": gif_path.name if gif_path is not None else None,
    }


def summarize_rollouts(task_results: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "success_rate": float(np.mean([1.0 if item["success"] else 0.0 for item in task_results])),
        "mean_steps": float(np.mean([item["steps"] for item in task_results])),
        "mean_rollout_mae": float(np.mean([item["mean_rollout_mae"] for item in task_results])),
    }


def render_report(report_path: Path, metrics: dict[str, Any], media_dir_name: str) -> None:
    checkpoints = metrics["checkpoints"]
    task_names = metrics["tasks"]

    def metric_card(label: str, value: str, tone: str) -> str:
        return f'<div class="metric-card {tone}"><div class="metric-label">{html.escape(label)}</div><div class="metric-value">{html.escape(value)}</div></div>'

    hero_cards: list[str] = []
    for checkpoint in checkpoints:
        overall = checkpoint["overall"]
        success_rate_text = f"{overall['success_rate'] * 100:.1f}%"
        rollout_mae_text = f"{overall['mean_rollout_mae']:.4f}"
        mean_steps_text = f"{overall['mean_steps']:.1f}"
        hero_cards.append(
            "<section class=\"hero-panel\">"
            f"<div class=\"eyebrow\">{html.escape(checkpoint['label'])}</div>"
            f"<h2>{html.escape(checkpoint['title'])}</h2>"
            "<div class=\"metric-grid\">"
            f"{metric_card('Success Rate', success_rate_text, 'warm')}"
            f"{metric_card('Mean Rollout MAE', rollout_mae_text, 'cool')}"
            f"{metric_card('Mean Steps', mean_steps_text, 'neutral')}"
            "</div>"
            "</section>"
        )

    comparison_rows: list[str] = []
    for task in task_names:
        values = []
        for checkpoint in checkpoints:
            task_metric = checkpoint["per_task"][task]
            values.append(
                f"<td><strong>{task_metric['success_rate'] * 100:.1f}%</strong><span>{task_metric['mean_rollout_mae']:.4f} mae</span></td>"
            )
        comparison_rows.append(f"<tr><th>{html.escape(task)}</th>{''.join(values)}</tr>")

    gallery_cards: list[str] = []
    final_checkpoint = checkpoints[-1]
    for task in task_names:
        demo = final_checkpoint["demos"][task]
        gif_target = f"{media_dir_name}/{html.escape(demo['gif'])}" if demo.get("gif") else ""
        gallery_cards.append(
            "<article class=\"demo-card\">"
            f"<div class=\"demo-header\"><span>{html.escape(task)}</span><span class=\"badge {'success' if demo['success'] else 'fail'}\">{'success' if demo['success'] else 'needs work'}</span></div>"
            f"<img src=\"{gif_target}\" alt=\"{html.escape(task)} demo\" />"
            f"<p>{html.escape(demo['instruction'])}</p>"
            f"<div class=\"demo-meta\">steps {demo['steps']} · mae {demo['mean_rollout_mae']:.4f}</div>"
            "</article>"
        )

    table_headers = "".join([f"<th>{html.escape(checkpoint['label'])}</th>" for checkpoint in checkpoints])

    html_text = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>VLA Evaluation Report</title>
  <style>
    :root {{
      --bg: #f4efe6;
      --panel: rgba(255, 251, 245, 0.88);
      --ink: #1d2730;
      --muted: #5d6a72;
      --warm: #d36a2e;
      --cool: #16697a;
      --line: rgba(29, 39, 48, 0.12);
      --shadow: 0 18px 50px rgba(55, 40, 18, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI Variable", "Trebuchet MS", "Avenir Next", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(211, 106, 46, 0.18), transparent 28%),
        radial-gradient(circle at bottom right, rgba(22, 105, 122, 0.18), transparent 30%),
        linear-gradient(180deg, #f8f3eb 0%, var(--bg) 100%);
    }}
    .shell {{ max-width: 1180px; margin: 0 auto; padding: 32px 20px 64px; }}
    .headline {{
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 20px;
      align-items: end;
      margin-bottom: 24px;
    }}
    .headline h1 {{ font-family: Georgia, "Times New Roman", serif; font-size: clamp(2.4rem, 5vw, 4.2rem); margin: 0; line-height: 0.95; }}
    .headline p {{ margin: 0; color: var(--muted); font-size: 1.05rem; line-height: 1.6; }}
    .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 24px; box-shadow: var(--shadow); }}
    .hero-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; margin-bottom: 18px; }}
    .hero-panel {{ padding: 22px; }}
    .eyebrow {{ text-transform: uppercase; letter-spacing: 0.12em; font-size: 0.72rem; color: var(--muted); margin-bottom: 10px; }}
    .hero-panel h2 {{ margin: 0 0 16px; font-size: 1.35rem; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }}
    .metric-card {{ border-radius: 18px; padding: 16px; min-height: 104px; display: flex; flex-direction: column; justify-content: space-between; }}
    .metric-card.warm {{ background: linear-gradient(160deg, rgba(211, 106, 46, 0.16), rgba(211, 106, 46, 0.05)); }}
    .metric-card.cool {{ background: linear-gradient(160deg, rgba(22, 105, 122, 0.16), rgba(22, 105, 122, 0.05)); }}
    .metric-card.neutral {{ background: linear-gradient(160deg, rgba(29, 39, 48, 0.08), rgba(29, 39, 48, 0.03)); }}
    .metric-label {{ color: var(--muted); font-size: 0.84rem; }}
    .metric-value {{ font-size: 1.7rem; font-weight: 700; }}
    .section {{ padding: 22px; margin-top: 18px; }}
    .section h3 {{ margin: 0 0 14px; font-size: 1.2rem; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 14px 12px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    td span {{ display: block; color: var(--muted); font-size: 0.88rem; margin-top: 4px; }}
    .demo-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }}
    .demo-card {{ background: rgba(255,255,255,0.6); border: 1px solid var(--line); border-radius: 20px; padding: 14px; }}
    .demo-card img {{ width: 100%; border-radius: 14px; border: 1px solid var(--line); background: #d7ccbb; display: block; }}
    .demo-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; font-weight: 700; }}
    .badge {{ border-radius: 999px; padding: 6px 10px; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; }}
    .badge.success {{ background: rgba(69, 140, 87, 0.16); color: #2b6d3d; }}
    .badge.fail {{ background: rgba(180, 74, 55, 0.16); color: #9a3d2d; }}
    .demo-card p {{ margin: 10px 0 6px; color: var(--muted); min-height: 2.8em; }}
    .demo-meta {{ font-size: 0.88rem; color: var(--ink); }}
    .footer {{ margin-top: 16px; color: var(--muted); font-size: 0.92rem; }}
    @media (max-width: 900px) {{
      .headline, .hero-grid, .demo-grid, .metric-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class=\"shell\">
    <section class=\"headline\">
      <div>
        <div class=\"eyebrow\">Baby Step 10</div>
        <h1>VLA Policy Evaluation Report</h1>
      </div>
      <p>Static report for GitHub-friendly publishing. It compares behavior-cloned and post-trained checkpoints, then shows rollout GIFs so readers can see what the robot is trying to do rather than only reading scalar metrics.</p>
    </section>
    <section class=\"hero-grid\">{''.join(hero_cards)}</section>
    <section class=\"panel section\">
      <h3>Per-Task Comparison</h3>
      <table>
        <thead><tr><th>Task</th>{table_headers}</tr></thead>
        <tbody>{''.join(comparison_rows)}</tbody>
      </table>
    </section>
    <section class=\"panel section\">
      <h3>Post-Training Demo Gallery</h3>
      <div class=\"demo-grid\">{''.join(gallery_cards)}</div>
      <div class=\"footer\">GIFs are generated from the final checkpoint so the repo can show qualitative behavior directly inside a browser or GitHub Pages site.</div>
    </section>
  </main>
</body>
</html>
"""
    report_path.write_text(html_text, encoding="utf-8")


def evaluate_checkpoint(
    label: str,
    title: str,
    checkpoint_path: Path | None,
    config: dict[str, Any],
    device: torch.device,
    media_dir: Path,
    save_demo_gifs: bool,
    per_task_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    # Build model cache: load shared model once, or per-task models lazily
    model_cache: dict[str, VLABehaviorCloningModel] = {}
    if checkpoint_path is not None:
        model_cache[str(checkpoint_path)] = load_policy(config["model"], checkpoint_path, device)

    eval_cfg = config["eval"]
    rng = np.random.default_rng(int(config["experiment"]["seed"]) + hash(label) % 1000)
    tasks = eval_cfg.get("tasks", TASKS)
    num_episodes_per_task = int(eval_cfg["num_episodes_per_task"])

    per_task: dict[str, dict[str, float]] = {}
    demos: dict[str, dict[str, Any]] = {}
    all_rollouts: list[dict[str, Any]] = []

    for task in tasks:
        # Determine which model to use for this task
        if per_task_paths and task in per_task_paths:
            task_ckpt_str = per_task_paths[task]
            if task_ckpt_str not in model_cache:
                model_cache[task_ckpt_str] = load_policy(config["model"], Path(task_ckpt_str), device)
            model = model_cache[task_ckpt_str]
        else:
            model = next(iter(model_cache.values()))

        task_rollouts: list[dict[str, Any]] = []
        for episode_idx in range(num_episodes_per_task):
            save_demo = save_demo_gifs and episode_idx == 0
            gif_path = media_dir / f"{label}_{task}.gif" if save_demo else None
            result = run_rollout(
                model=model,
                task=task,
                rng=rng,
                device=device,
                eval_cfg=eval_cfg,
                model_cfg=config["model"],
                save_demo=save_demo,
                gif_path=gif_path,
            )
            task_rollouts.append(result)
            all_rollouts.append(result)
            if episode_idx == 0:
                demos[task] = result

        per_task[task] = summarize_rollouts(task_rollouts)

    overall = summarize_rollouts(all_rollouts)
    ckpt_summary = str(checkpoint_path) if checkpoint_path else str(per_task_paths)
    return {
        "label": label,
        "title": title,
        "checkpoint": ckpt_summary,
        "overall": overall,
        "per_task": per_task,
        "demos": demos,
    }


def train(config: dict[str, Any]) -> None:
    set_seed(int(config["experiment"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(config["experiment"]["output_dir"])
    media_dir = out_dir / "media"
    out_dir.mkdir(parents=True, exist_ok=True)
    media_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_cfg = config["checkpoints"]
    checkpoint_results = []
    for item in checkpoints_cfg:
        raw_per_task = item.get("per_task_paths", None)
        per_task_paths: dict[str, str] | None = dict(raw_per_task) if raw_per_task else None
        raw_path = item.get("path", None)
        checkpoint_path: Path | None = Path(str(raw_path)) if raw_path else None
        checkpoint_results.append(
            evaluate_checkpoint(
                label=str(item["label"]),
                title=str(item["title"]),
                checkpoint_path=checkpoint_path,
                config=config,
                device=device,
                media_dir=media_dir,
                save_demo_gifs=bool(config["eval"]["save_demo_gifs"]),
                per_task_paths=per_task_paths,
            )
        )

    metrics = {
        "tasks": config["eval"].get("tasks", TASKS),
        "checkpoints": checkpoint_results,
    }

    metrics_path = out_dir / "metrics.json"
    report_path = out_dir / "report.html"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    render_report(report_path, metrics, media_dir_name="media")

    print("\n[Step 10] Evaluation complete")
    print("=" * 72)
    for checkpoint in checkpoint_results:
        overall = checkpoint["overall"]
        print(
            f"{checkpoint['label']:>10s} | "
            f"success={overall['success_rate'] * 100:5.1f}% | "
            f"mean_mae={overall['mean_rollout_mae']:.4f} | "
            f"mean_steps={overall['mean_steps']:.1f}"
        )
    print(f"Metrics JSON : {metrics_path}")
    print(f"HTML report  : {report_path}")
    print(f"Demo media   : {media_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage D evaluation and visual demo report")
    parser.add_argument("--config", type=str, default="configs/eval.yaml", help="Path to evaluation YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()