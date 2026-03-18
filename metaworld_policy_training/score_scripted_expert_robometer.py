"""
Score a scripted expert's successful trajectory using Robometer (HTTP server).

Evaluates the Robometer reward signal itself, independent of any trained RL policy.
Uses MetaWorld's built-in scripted policies to generate guaranteed-success trajectories.

Generates:
  1. correlation_analysis.txt  – Pearson/Spearman between Robometer outputs and GT reward
  2. trajectory_analysis.mp4   – 2x2 video (env | raw progress | diff progress | GT reward)

Usage (from metaworld_policy_training/):
    python score_scripted_expert_robometer.py \
        --server_url http://<hostname>:8002 \
        --output_dir score_output/scripted_expert_robometer
"""

import os
import sys
import io
import base64
import functools
import argparse
import numpy as np
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "Metaworld"))

from envs.metaworld import environment_to_instruction
from tests.metaworld.envs.mujoco.sawyer_xyz.test_scripted_policies import ALL_ENVS, test_cases_latest_nonoise

RESOLUTION = (640, 480)
CAMERA = "corner2"

DEFAULT_ENVS = [
    "window-close-v2",
    "reach-wall-v2",
    "faucet-close-v2",
    "coffee-button-v2",
    "button-press-wall-v2",
    "door-lock-v2",
    "handle-press-side-v2",
    "sweep-into-v2",
]


def run_scripted_expert(env_name, seed=0, max_attempts=15):
    """Run scripted expert policy until a successful trajectory is found."""
    policy = functools.reduce(
        lambda a, b: a if a[0] == env_name else b, test_cases_latest_nonoise
    )[1]

    env = ALL_ENVS[env_name]()
    env._partially_observable = False
    env._freeze_rand_vec = False
    env._set_task_called = True

    for attempt in range(max_attempts):
        env._partially_observable = False
        env._freeze_rand_vec = False
        env._set_task_called = True
        env.seed(seed + attempt)
        env.reset()
        env.reset_model()
        o = env.reset()

        imgs = []
        gt_rewards = []
        success = False
        success_step = None

        for step in range(env.max_path_length):
            img = env.sim.render(*RESOLUTION, mode="offscreen", camera_name=CAMERA).astype(np.uint8)
            imgs.append(img)

            a = policy.get_action(o)
            a = np.clip(a, env.action_space.low, env.action_space.high)
            o, r, done, info = env.step(a)
            gt_rewards.append(r)

            if info["success"] and not success:
                success = True
                success_step = step
                img = env.sim.render(*RESOLUTION, mode="offscreen", camera_name=CAMERA).astype(np.uint8)
                imgs.append(img)
                gt_rewards.append(r)
                break

        if success:
            print(f"Scripted expert succeeded at step {success_step} (attempt {attempt}, seed {seed + attempt})")
            return imgs, np.array(gt_rewards), success, success_step

    print(f"WARNING: scripted expert failed to succeed in {max_attempts} attempts")
    return imgs, np.array(gt_rewards), False, None


# ── Robometer server scoring ──

def score_trajectory_server(server_url, raw_images, task_text, max_frames=16):
    """Score a trajectory by calling the Robometer HTTP server.

    Sends all frames (subsampled to max_frames) and returns per-frame progress.
    """
    frames = np.stack(raw_images, axis=0)  # (T, H, W, C)
    T = frames.shape[0]
    if T > max_frames:
        indices = np.linspace(0, T - 1, max_frames, dtype=int)
        frames_sub = frames[indices]
    else:
        frames_sub = frames
        indices = np.arange(T)

    buf = io.BytesIO()
    np.save(buf, frames_sub)
    frames_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    payload = {
        "frames_b64": frames_b64,
        "task": task_text,
        "sample_type": "progress",
    }

    resp = requests.post(f"{server_url}/predict", json=payload, timeout=120)
    resp.raise_for_status()
    result = resp.json()
    progress = result.get("progress", [0.0])
    if not isinstance(progress, list):
        progress = [progress]

    progress = np.array(progress, dtype=np.float64)
    if T > max_frames:
        progress_full = np.interp(np.arange(T), indices, progress[:len(indices)])
    else:
        progress_full = progress[:T]

    return progress_full


# ── Correlation analysis ──

def compute_correlations(x, y, label_x, label_y):
    if len(x) < 3:
        return {"pearson": float("nan"), "p_pearson": float("nan"),
                "spearman": float("nan"), "label": f"{label_x} vs {label_y}", "n": len(x)}
    p_r, p_p = pearsonr(x, y)
    s_r, _ = spearmanr(x, y)
    return {"pearson": p_r, "p_pearson": p_p, "spearman": s_r,
            "label": f"{label_x} vs {label_y}", "n": len(x)}


def write_correlation_report(path, env_id, results, success, success_step, num_frames):
    with open(path, "w") as f:
        f.write(f"Correlation Analysis (Scripted Expert + Robometer) for {env_id}\n")
        f.write(f"Episode: {num_frames} frames, success={success}, success_step={success_step}\n\n")
        for r in results:
            f.write(f"--- {r['label']} (n={r['n']}) ---\n")
            f.write(f"  Pearson:  {r['pearson']:.6f} (p={r['p_pearson']:.2e})\n")
            f.write(f"  Spearman: {r['spearman']:.6f}\n\n")
    print(f"Saved correlation analysis to {path}")


# ── Video generation ──

def generate_video(images, progress_raw, progress_diff, gt_rewards,
                   video_path, env_id, success_step, fps=20):
    diff_padded = np.concatenate([[0.0], progress_diff])
    num_frames = len(images)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax_img = axes[0, 0]
    im = ax_img.imshow(images[0])
    ax_img.set_title("Environment (Scripted Expert)", fontsize=12)
    ax_img.axis("off")
    step_text = ax_img.text(
        0.02, 0.98, "", transform=ax_img.transAxes, fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax_raw = axes[0, 1]
    (line_raw,) = ax_raw.plot([], [], "b-", linewidth=2)
    ax_raw.set_xlim(0, num_frames)
    margin = max(0.05, (np.max(progress_raw) - np.min(progress_raw)) * 0.1)
    ax_raw.set_ylim(np.min(progress_raw) - margin, np.max(progress_raw) + margin)
    ax_raw.set_xlabel("Step")
    ax_raw.set_ylabel("Progress")
    ax_raw.set_title("Robometer Raw Progress", fontsize=12)
    ax_raw.grid(True, alpha=0.3)
    (dot_raw,) = ax_raw.plot([], [], "bo", markersize=5)

    ax_diff = axes[1, 0]
    (line_diff,) = ax_diff.plot([], [], "m-", linewidth=2)
    ax_diff.set_xlim(0, num_frames)
    d_margin = max(0.01, (np.max(diff_padded) - np.min(diff_padded)) * 0.15)
    ax_diff.set_ylim(np.min(diff_padded) - d_margin, np.max(diff_padded) + d_margin)
    ax_diff.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax_diff.set_xlabel("Step")
    ax_diff.set_ylabel("Diff")
    ax_diff.set_title("Robometer Diff (P(s')-P(s))", fontsize=12)
    ax_diff.grid(True, alpha=0.3)
    (dot_diff,) = ax_diff.plot([], [], "mo", markersize=5)

    ax_gt = axes[1, 1]
    (line_gt,) = ax_gt.plot([], [], "r-", linewidth=2)
    ax_gt.set_xlim(0, num_frames)
    g_margin = max(0.1, (np.max(gt_rewards) - np.min(gt_rewards)) * 0.1)
    ax_gt.set_ylim(np.min(gt_rewards) - g_margin, np.max(gt_rewards) + g_margin)
    ax_gt.set_xlabel("Step")
    ax_gt.set_ylabel("GT Reward")
    ax_gt.set_title("GT Reward", fontsize=12)
    ax_gt.grid(True, alpha=0.3)
    (dot_gt,) = ax_gt.plot([], [], "ro", markersize=5)

    plt.suptitle(f"Scripted Expert Trajectory Analysis (Robometer) - {env_id}", fontsize=14)
    plt.tight_layout()

    def init():
        for ln in [line_raw, line_diff, line_gt]:
            ln.set_data([], [])
        for d in [dot_raw, dot_diff, dot_gt]:
            d.set_data([], [])
        step_text.set_text("")
        return line_raw, line_diff, line_gt, dot_raw, dot_diff, dot_gt, step_text, im

    def animate(frame):
        im.set_array(images[frame])
        status = " SUCCESS!" if success_step is not None and frame >= success_step else ""
        step_text.set_text(f"Step: {frame}/{num_frames - 1}{status}")
        x = np.arange(frame + 1)
        line_raw.set_data(x, progress_raw[: frame + 1])
        line_diff.set_data(x, diff_padded[: frame + 1])
        line_gt.set_data(x, gt_rewards[: frame + 1])
        dot_raw.set_data([frame], [progress_raw[frame]])
        dot_diff.set_data([frame], [diff_padded[frame]])
        dot_gt.set_data([frame], [gt_rewards[frame]])
        return line_raw, line_diff, line_gt, dot_raw, dot_diff, dot_gt, step_text, im

    anim = FuncAnimation(fig, animate, init_func=init, frames=num_frames, interval=50, blit=True)
    writer = FFMpegWriter(fps=fps, bitrate=2400)
    print(f"Saving video to {video_path} ({num_frames} frames)...")
    anim.save(video_path, writer=writer)
    plt.close(fig)
    print(f"Saved video to {video_path}")


# ── Main ──

def score_one_env(env_id, server_url, max_frames, seed, output_dir, fps):
    text_instruction = environment_to_instruction[env_id]
    env_output_dir = os.path.join(output_dir, env_id)
    os.makedirs(env_output_dir, exist_ok=True)

    print(f"\nEnvironment: {env_id}")
    print(f"Instruction: {text_instruction}")

    print("  Running scripted expert...")
    raw_images, gt_rewards, success, success_step = run_scripted_expert(
        env_id, seed=seed
    )
    num_frames = len(raw_images)
    print(f"  Episode: {num_frames} frames, {len(gt_rewards)} rewards, success={success}")

    print("  Scoring trajectory with Robometer...")
    progress_raw = score_trajectory_server(
        server_url, raw_images, text_instruction, max_frames=max_frames,
    )
    progress_diff = np.diff(progress_raw)

    print(f"  progress_raw length: {len(progress_raw)}, gt_rewards length: {len(gt_rewards)}")

    print("  Computing correlations...")
    results = []

    results.append(compute_correlations(
        progress_raw, gt_rewards, "Raw Progress", "GT Reward"))

    results.append(compute_correlations(
        progress_diff, gt_rewards[1:], "Diff Progress", "GT Reward"))

    n30 = min(30, num_frames)
    results.append(compute_correlations(
        progress_raw[:n30], gt_rewards[:n30], "Raw Progress (first 30)", "GT Reward (first 30)"))

    n30d = min(30, len(progress_diff))
    results.append(compute_correlations(
        progress_diff[:n30d], gt_rewards[1:n30d + 1], "Diff Progress (first 30)", "GT Reward (first 30)"))

    if success_step is not None and success_step > 2:
        results.append(compute_correlations(
            progress_raw[:success_step + 1], gt_rewards[:success_step + 1],
            "Raw Progress (pre-success)", "GT Reward (pre-success)"))
        results.append(compute_correlations(
            progress_diff[:success_step], gt_rewards[1:success_step + 1],
            "Diff Progress (pre-success)", "GT Reward (pre-success)"))

    for r in results:
        print(f"    {r['label']} (n={r['n']}): Pearson={r['pearson']:.4f}, Spearman={r['spearman']:.4f}")

    corr_path = os.path.join(env_output_dir, "correlation_analysis.txt")
    write_correlation_report(corr_path, env_id, results, success, success_step, num_frames)

    print("  Generating video...")
    video_path = os.path.join(env_output_dir, "trajectory_analysis.mp4")
    generate_video(raw_images, progress_raw, progress_diff, gt_rewards,
                   video_path, env_id, success_step, fps=fps)

    print(f"  Done: {env_id}")


def main():
    parser = argparse.ArgumentParser(description="Score scripted expert trajectory with Robometer")
    parser.add_argument("--env_id", type=str, nargs="*", default=None,
                        help="One or more env IDs. If omitted, runs all 8 default envs.")
    parser.add_argument("--server_url", type=str, required=True,
                        help="Robometer server URL (e.g. http://b17-15.hpc.usc.edu:8002)")
    parser.add_argument("--max_frames", type=int, default=16,
                        help="Max frames to send to Robometer per request")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="score_output/scripted_expert_robometer")
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    env_ids = args.env_id if args.env_id else DEFAULT_ENVS
    print(f"Will score {len(env_ids)} environment(s): {env_ids}")
    print(f"Robometer server: {args.server_url}")

    # Health check
    try:
        resp = requests.get(f"{args.server_url}/health", timeout=10)
        resp.raise_for_status()
        print(f"Robometer server is healthy: {resp.json()}\n")
    except Exception as e:
        print(f"ERROR: Robometer server health check failed: {e}")
        sys.exit(1)

    for i, env_id in enumerate(env_ids):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(env_ids)}] {env_id}")
        print(f"{'='*60}")
        score_one_env(env_id, args.server_url, args.max_frames,
                      args.seed, args.output_dir, args.fps)

    print(f"\nAll done! Results in {args.output_dir}/")


if __name__ == "__main__":
    main()
