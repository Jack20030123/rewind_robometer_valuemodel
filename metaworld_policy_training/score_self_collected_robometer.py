"""
Score self-collected videos AND example_videos using a Robometer HTTP server.

Reads existing mp4/gif videos, sends frames to a running Robometer server,
and generates 1x3 visualisation videos:
  left: original video | middle: raw progress curve | right: diff progress curve

Usage (from metaworld_policy_training/):
    python score_self_collected_robometer.py \
        --video_root ../self_collected_videos \
        --example_video_dir ../example_videos \
        --server_url http://<hostname>:8002 \
        --output_dir score_output/self_collected_robometer
"""

import os
import sys
import io
import base64
import argparse
import numpy as np
import requests
import imageio.v3 as iio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Inline dict to avoid importing metaworld (which requires mujoco)
environment_to_instruction = {
    "assembly-v2": "assembly",
    "basketball-v2": "play basketball",
    "bin-picking-v2": "pick bin",
    "box-close-v2": "closing box",
    "button-press-topdown-v2": "Press the button from top",
    "button-press-topdown-wall-v2": "Press the button from top",
    "button-press-v2": "Press the button from side",
    "button-press-wall-v2": "Press the button from side",
    "coffee-button-v2": "Press the coffee button",
    "coffee-pull-v2": "Pull the coffee cup",
    "coffee-push-v2": "Push the coffee cup",
    "dial-turn-v2": "Turn the dial",
    "disassemble-v2": "disassemble",
    "door-close-v2": "Close the door",
    "door-lock-v2": "Turn door lock counter-clockwise",
    "door-open-v2": "Open the door",
    "door-unlock-v2": "Turn door lock clockwise",
    "hand-insert-v2": "Pick up the block and insert it into the hole",
    "drawer-close-v2": "Close the drawer",
    "drawer-open-v2": "open drawer",
    "faucet-open-v2": "Open the faucet",
    "faucet-close-v2": "Close the faucet",
    "hammer-v2": "hammer nail",
    "handle-press-side-v2": "Press the handle from side",
    "handle-press-v2": "Press the handle",
    "handle-pull-side-v2": "Pull the handle up from the side",
    "handle-pull-v2": "Pull the handle",
    "lever-pull-v2": "pull lever",
    "peg-insert-side-v2": "Insert the peg",
    "pick-place-wall-v2": "Pick up the block and placing it to the goal position",
    "pick-out-of-hole-v2": "pick bin",
    "reach-v2": "Reach the goal",
    "push-back-v2": "Push the block back to the goal",
    "push-v2": "Push the block to the goal",
    "pick-place-v2": "Pick up the block and placing it to the goal position",
    "plate-slide-v2": "Slide the plate into the gate",
    "plate-slide-side-v2": "Slide the plate into the gate from the side",
    "plate-slide-back-v2": "Slide the plate out of the gate",
    "plate-slide-back-side-v2": "Slide the plate out of the gate from the side",
    "peg-unplug-side-v2": "unplug peg",
    "soccer-v2": "Slide the ball into the gate",
    "stick-push-v2": "Push the stick",
    "stick-pull-v2": "Pull the stick",
    "push-wall-v2": "push bin",
    "reach-wall-v2": "Reach the goal",
    "shelf-place-v2": "place bin to shelf",
    "sweep-into-v2": "Sweep the block into the hole",
    "sweep-v2": "sweep block",
    "window-open-v2": "Open the window",
    "window-close-v2": "Close the window",
}


# -- Task descriptions for example_videos (not MetaWorld envs) --
EXAMPLE_VIDEO_TASKS = {
    "berkeley_rpt_stack_cup": "Pick up the yellow cup and stack it on the other cup",
    "jaco_play_pick_up_green_cup": "pick up the green cup",
    "soar_put_green_stick_in_brown_bowl": "Put green stick in brown bowl",
}


def dir_name_to_env_id(name):
    """Convert underscore directory name to MetaWorld v2 env id."""
    if name.endswith("-v2"):
        return name
    return name.replace("_", "-") + "-v2"


def read_video_frames(video_path):
    """Read video file and return list of RGB numpy arrays (H, W, 3)."""
    frames = iio.imread(video_path)
    result = []
    for frame in frames:
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        elif frame.shape[-1] == 4:
            frame = frame[:, :, :3]
        result.append(frame.astype(np.uint8))
    return result


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

    resp = requests.post(f"{server_url}/predict", json=payload, timeout=60)
    resp.raise_for_status()
    result = resp.json()
    progress = result.get("progress", [0.0])
    if not isinstance(progress, list):
        progress = [progress]

    # Interpolate back to original frame count if subsampled
    progress = np.array(progress, dtype=np.float64)
    if T > max_frames:
        progress_full = np.interp(np.arange(T), indices, progress[:len(indices)])
    else:
        progress_full = progress[:T]

    return progress_full


def generate_video(images, progress_raw, progress_diff_099, progress_diff_0999, video_path, title, fps=10):
    """Generate 2x2 MP4: top-left=video, top-right=raw progress, bottom-left=0.99 diff, bottom-right=0.999 diff."""
    diff_099_padded = np.concatenate([[0.0], progress_diff_099])
    diff_0999_padded = np.concatenate([[0.0], progress_diff_0999])
    num_frames = len(images)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Top-left: video frames
    ax_img = axes[0, 0]
    im = ax_img.imshow(images[0])
    ax_img.set_title("Video", fontsize=12)
    ax_img.axis("off")
    step_text = ax_img.text(
        0.02, 0.98, "", transform=ax_img.transAxes, fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Top-right: raw progress
    ax_raw = axes[0, 1]
    (line_raw,) = ax_raw.plot([], [], "b-", linewidth=2)
    ax_raw.set_xlim(0, num_frames)
    margin = max(0.05, (np.max(progress_raw) - np.min(progress_raw)) * 0.1)
    ax_raw.set_ylim(np.min(progress_raw) - margin, np.max(progress_raw) + margin)
    ax_raw.set_xlabel("Step")
    ax_raw.set_ylabel("Progress")
    ax_raw.set_title("Raw Progress (Robometer)", fontsize=12)
    ax_raw.grid(True, alpha=0.3)
    (dot_raw,) = ax_raw.plot([], [], "bo", markersize=5)

    # Bottom-left: 0.99 diff
    ax_d099 = axes[1, 0]
    (line_d099,) = ax_d099.plot([], [], "m-", linewidth=2)
    ax_d099.set_xlim(0, num_frames)
    d099_margin = max(0.01, (np.max(diff_099_padded) - np.min(diff_099_padded)) * 0.15)
    ax_d099.set_ylim(np.min(diff_099_padded) - d099_margin, np.max(diff_099_padded) + d099_margin)
    ax_d099.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax_d099.set_xlabel("Step")
    ax_d099.set_ylabel("Diff")
    ax_d099.set_title("0.99·P(s') - P(s)", fontsize=12)
    ax_d099.grid(True, alpha=0.3)
    (dot_d099,) = ax_d099.plot([], [], "mo", markersize=5)

    # Bottom-right: 0.999 diff
    ax_d0999 = axes[1, 1]
    (line_d0999,) = ax_d0999.plot([], [], "g-", linewidth=2)
    ax_d0999.set_xlim(0, num_frames)
    d0999_margin = max(0.01, (np.max(diff_0999_padded) - np.min(diff_0999_padded)) * 0.15)
    ax_d0999.set_ylim(np.min(diff_0999_padded) - d0999_margin, np.max(diff_0999_padded) + d0999_margin)
    ax_d0999.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax_d0999.set_xlabel("Step")
    ax_d0999.set_ylabel("Diff")
    ax_d0999.set_title("0.999·P(s') - P(s)", fontsize=12)
    ax_d0999.grid(True, alpha=0.3)
    (dot_d0999,) = ax_d0999.plot([], [], "go", markersize=5)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()

    def init():
        line_raw.set_data([], [])
        line_d099.set_data([], [])
        line_d0999.set_data([], [])
        dot_raw.set_data([], [])
        dot_d099.set_data([], [])
        dot_d0999.set_data([], [])
        step_text.set_text("")
        return line_raw, line_d099, line_d0999, dot_raw, dot_d099, dot_d0999, step_text, im

    def animate(frame):
        im.set_array(images[frame])
        step_text.set_text(f"Step: {frame}/{num_frames - 1}")
        x = np.arange(frame + 1)
        line_raw.set_data(x, progress_raw[: frame + 1])
        line_d099.set_data(x, diff_099_padded[: frame + 1])
        line_d0999.set_data(x, diff_0999_padded[: frame + 1])
        dot_raw.set_data([frame], [progress_raw[frame]])
        dot_d099.set_data([frame], [diff_099_padded[frame]])
        dot_d0999.set_data([frame], [diff_0999_padded[frame]])
        return line_raw, line_d099, line_d0999, dot_raw, dot_d099, dot_d0999, step_text, im

    anim = FuncAnimation(fig, animate, init_func=init, frames=num_frames, interval=50, blit=True)
    writer = FFMpegWriter(fps=fps, bitrate=2400)
    anim.save(video_path, writer=writer)
    plt.close(fig)


def collect_videos(video_root):
    """Walk video_root and yield (video_path, env_id, category) tuples."""
    SKIP_DIRS = {"eval_tasks", "train_tasks"}
    for entry in sorted(os.listdir(video_root)):
        entry_path = os.path.join(video_root, entry)
        if not os.path.isdir(entry_path):
            continue

        if entry in SKIP_DIRS:
            for env_dir in sorted(os.listdir(entry_path)):
                env_dir_path = os.path.join(entry_path, env_dir)
                if not os.path.isdir(env_dir_path):
                    continue
                env_id = dir_name_to_env_id(env_dir)
                for cat in sorted(os.listdir(env_dir_path)):
                    cat_path = os.path.join(env_dir_path, cat)
                    if not os.path.isdir(cat_path):
                        continue
                    for vf in sorted(os.listdir(cat_path)):
                        if vf.lower().endswith((".mp4", ".gif")):
                            yield os.path.join(cat_path, vf), env_id, f"{entry}/{env_dir}/{cat}"
        else:
            env_id = dir_name_to_env_id(entry)
            for cat in sorted(os.listdir(entry_path)):
                cat_path = os.path.join(entry_path, cat)
                if not os.path.isdir(cat_path):
                    continue
                for vf in sorted(os.listdir(cat_path)):
                    if vf.lower().endswith((".mp4", ".gif")):
                        yield os.path.join(cat_path, vf), env_id, f"{entry}/{cat}"


def collect_example_videos(example_dir):
    """Yield (video_path, task_text, category) for example_videos/."""
    if not os.path.isdir(example_dir):
        return
    for fname in sorted(os.listdir(example_dir)):
        if not fname.lower().endswith(".mp4"):
            continue
        stem = os.path.splitext(fname)[0]
        if stem in EXAMPLE_VIDEO_TASKS:
            yield (
                os.path.join(example_dir, fname),
                EXAMPLE_VIDEO_TASKS[stem],
                f"example_videos/{stem}",
            )


def main():
    parser = argparse.ArgumentParser(description="Score videos with Robometer server")
    parser.add_argument("--video_root", type=str, default="../self_collected_videos",
                        help="Root directory containing self-collected videos")
    parser.add_argument("--example_video_dir", type=str, default="../example_videos",
                        help="Directory containing example videos")
    parser.add_argument("--server_url", type=str, required=True,
                        help="Robometer server URL (e.g. http://b17-15.hpc.usc.edu:8002)")
    parser.add_argument("--output_dir", type=str, default="score_output/self_collected_robometer")
    parser.add_argument("--max_frames", type=int, default=16,
                        help="Max frames to send to Robometer per video")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    # Health check
    print(f"=== Checking Robometer server at {args.server_url} ===")
    try:
        r = requests.get(f"{args.server_url}/health", timeout=10)
        r.raise_for_status()
        print(f"Server OK: {r.json()}")
    except Exception as e:
        print(f"ERROR: Cannot reach server: {e}")
        sys.exit(1)

    # --- Collect self-collected videos ---
    all_jobs = []  # list of (video_path, task_text, rel_category)
    videos = list(collect_videos(args.video_root))
    print(f"\nFound {len(videos)} self-collected videos.")
    for video_path, env_id, rel_category in videos:
        if env_id not in environment_to_instruction:
            print(f"  SKIP unknown env_id '{env_id}': {video_path}")
            continue
        task_text = environment_to_instruction[env_id]
        all_jobs.append((video_path, task_text, rel_category))

    # --- Collect example videos ---
    example_videos = list(collect_example_videos(args.example_video_dir))
    print(f"Found {len(example_videos)} example videos.")
    for video_path, task_text, rel_category in example_videos:
        all_jobs.append((video_path, task_text, rel_category))

    print(f"\nTotal: {len(all_jobs)} videos to score.\n")

    for idx, (video_path, task_text, rel_category) in enumerate(all_jobs):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"[{idx+1}/{len(all_jobs)}] {rel_category}/{video_name}  task=\"{task_text}\"")

        # Read frames
        try:
            raw_images = read_video_frames(video_path)
        except Exception as e:
            print(f"  ERROR reading video: {e}")
            continue

        if len(raw_images) < 2:
            print(f"  SKIP — only {len(raw_images)} frame(s)")
            continue

        num_frames = len(raw_images)

        # Score via server
        try:
            progress_raw = score_trajectory_server(
                args.server_url, raw_images, task_text, max_frames=args.max_frames
            )
        except Exception as e:
            print(f"  ERROR scoring: {e}")
            continue

        progress_diff_099 = 0.99 * progress_raw[1:] - progress_raw[:-1]
        progress_diff_0999 = 0.999 * progress_raw[1:] - progress_raw[:-1]

        # Output path
        out_dir = os.path.join(args.output_dir, rel_category)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{video_name}_scored.mp4")

        title = f"{rel_category}/{video_name} [Robometer]"
        generate_video(raw_images, progress_raw, progress_diff_099, progress_diff_0999, out_path, title, fps=args.fps)
        print(f"  -> {out_path}  ({num_frames} frames, progress [{progress_raw.min():.3f}, {progress_raw.max():.3f}])")

    print("\nDone!")


if __name__ == "__main__":
    main()
