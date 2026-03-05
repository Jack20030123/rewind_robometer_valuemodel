"""
Label MetaWorld trajectories with Robometer progress rewards.

Produces an h5 file with the same structure as metaworld_label_reward.py output,
but with rewards computed by Robometer instead of ReWiND.

Usage:
  # Server mode (recommended — no 4B model loaded locally):
  python robometer_label_reward.py --use_server --server_url http://gpu-node:8000

  # Local mode (loads Robometer-4B locally, needs large GPU):
  python robometer_label_reward.py --model_path aliangdw/Robometer-4B

  # Progress diff mode:
  python robometer_label_reward.py --use_server --use_progress_diff \
      --output_path datasets/metaworld_labeled_robometer_diff.h5
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import h5py
import torch
import argparse
import numpy as np
from tqdm import tqdm

from utils.processing_utils import dino_load_image


# ---------------------------------------------------------------------------
# Environment → task text mapping (same as metaworld_policy_training/envs/metaworld.py)
# ---------------------------------------------------------------------------
ENVIRONMENT_TO_INSTRUCTION = {
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


# ---------------------------------------------------------------------------
# DINO embeddings (needed for policy observation in offline training)
# ---------------------------------------------------------------------------
DINO_BATCH_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dino():
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", force_reload=False)
    return model.to(device)


def get_dino_embeddings(imgs_list, dino_model):
    episode_images_dino = [dino_load_image(img) for img in imgs_list]
    episode_images_dino = [
        torch.concatenate(episode_images_dino[i : i + DINO_BATCH_SIZE])
        for i in range(0, len(episode_images_dino), DINO_BATCH_SIZE)
    ]
    embedding_list = []
    for batch in episode_images_dino:
        emb = dino_model(batch.to(device)).squeeze().detach().cpu().numpy()
        if emb.ndim == 1:
            emb = np.expand_dims(emb, 0)
        embedding_list.append(emb)
    return np.concatenate(embedding_list)


# ---------------------------------------------------------------------------
# Robometer inference — per-step approach
# At step t, feed frames[0:t+1] (subsampled to max_frames), get last-frame progress.
# This matches online training behavior exactly.
# ---------------------------------------------------------------------------
def robometer_progress_per_step_local(
    frames, task_text, model, tokenizer, batch_collator, exp_config, max_frames=4
):
    """
    Compute per-frame progress using local Robometer model.

    Args:
        frames: (F, H, W, C) uint8, F = num_steps + 1
        task_text: task description string

    Returns:
        np.ndarray of shape (F,) with progress values for each frame position.
    """
    from robometer.data.dataset_types import ProgressSample, Trajectory
    from robometer.evals.eval_server import compute_batch_outputs

    loss_config = getattr(exp_config, "loss", None)
    is_discrete = (
        getattr(loss_config, "progress_loss_type", "l2").lower() == "discrete"
        if loss_config
        else False
    )
    num_bins = (
        getattr(loss_config, "progress_discrete_bins", None)
        or getattr(exp_config.model, "progress_discrete_bins", 10)
    )

    F = frames.shape[0]
    progress_values = np.zeros(F, dtype=np.float32)

    for t in range(F):
        sub_frames = frames[: t + 1]  # frames[0:t+1]
        T = sub_frames.shape[0]

        # Subsample to max_frames (same as online wrapper)
        if T > max_frames:
            indices = np.linspace(0, T - 1, max_frames, dtype=int)
            sub_frames = sub_frames[indices]
            T = max_frames

        traj = Trajectory(
            frames=sub_frames,
            frames_shape=tuple(sub_frames.shape),
            task=task_text,
            id="0",
            metadata={"subsequence_length": T},
            video_embeddings=None,
        )
        sample = ProgressSample(trajectory=traj, sample_type="progress")
        batch = batch_collator([sample])

        progress_inputs = batch["progress_inputs"]
        for key, value in progress_inputs.items():
            if hasattr(value, "to"):
                progress_inputs[key] = value.to(device)

        with torch.no_grad():
            results = compute_batch_outputs(
                model, tokenizer, progress_inputs,
                sample_type="progress",
                is_discrete_mode=is_discrete,
                num_bins=num_bins,
            )

        preds = results.get("progress_pred", [])
        if preds and len(preds) > 0:
            progress_values[t] = float(preds[0][-1])

    return progress_values


def robometer_progress_per_step_server(frames, task_text, server_url, max_frames=4):
    """
    Compute per-frame progress using Robometer HTTP server.

    Args:
        frames: (F, H, W, C) uint8
        task_text: task description string
        server_url: e.g. "http://localhost:8000"

    Returns:
        np.ndarray of shape (F,) with progress values.
    """
    import requests
    import io
    import base64

    F = frames.shape[0]
    progress_values = np.zeros(F, dtype=np.float32)

    for t in range(F):
        sub_frames = frames[: t + 1]
        T = sub_frames.shape[0]

        if T > max_frames:
            indices = np.linspace(0, T - 1, max_frames, dtype=int)
            sub_frames = sub_frames[indices]

        buf = io.BytesIO()
        np.save(buf, sub_frames)
        frames_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        payload = {
            "frames_b64": frames_b64,
            "task": task_text,
            "sample_type": "progress",
        }

        try:
            resp = requests.post(
                f"{server_url.rstrip('/')}/predict", json=payload, timeout=60,
            )
            resp.raise_for_status()
            result = resp.json()
            progress = result.get("progress", result.get("reward", 0.0))
            if isinstance(progress, list):
                progress_values[t] = float(progress[-1])
            else:
                progress_values[t] = float(progress)
        except Exception as e:
            print(f"  [WARNING] Server call failed at step {t}: {e}")
            progress_values[t] = 0.0

    return progress_values


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Label MetaWorld trajectories with Robometer rewards."
    )
    parser.add_argument(
        "--h5_video_path",
        default="datasets/metaworld_generation.h5",
        help="Source trajectories (raw video frames).",
    )
    parser.add_argument(
        "--h5_embedding_path",
        default="datasets/metaworld_embeddings_train.h5",
        help="Language annotation embeddings (MiniLM).",
    )
    parser.add_argument(
        "--output_path",
        default="datasets/metaworld_labeled_robometer.h5",
        help="Output labeled dataset.",
    )
    parser.add_argument(
        "--use_progress_diff",
        action="store_true",
        help="Use reward = P(s') - P(s) instead of P(s).",
    )
    parser.add_argument(
        "--use_server",
        action="store_true",
        help="Use Robometer HTTP server instead of loading model locally.",
    )
    parser.add_argument(
        "--server_url",
        default="http://localhost:8000",
        help="Robometer server URL (only used with --use_server).",
    )
    parser.add_argument(
        "--model_path",
        default="aliangdw/Robometer-4B",
        help="HuggingFace model ID or local path (only used without --use_server).",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=4,
        help="Max frames per Robometer inference call.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for DINO and Robometer (local mode).",
    )
    args = parser.parse_args()

    global device
    device = torch.device(args.device)

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    print("Loading DINO for image embeddings...")
    dino_model = load_dino()

    robometer_model = robometer_tokenizer = robometer_collator = robometer_exp_config = None
    if not args.use_server:
        print(f"Loading Robometer from {args.model_path} ...")
        from robometer.utils.save import load_model_from_hf
        from robometer.utils.setup_utils import setup_batch_collator

        exp_config, tokenizer, processor, model = load_model_from_hf(
            model_path=args.model_path, device=device,
        )
        model.eval()
        robometer_model = model
        robometer_tokenizer = tokenizer
        robometer_exp_config = exp_config
        robometer_collator = setup_batch_collator(
            processor, tokenizer, exp_config, is_eval=True
        )
        print("Robometer loaded.")
    else:
        print(f"Using Robometer server at {args.server_url}")

    # ------------------------------------------------------------------
    # Open source h5 files
    # ------------------------------------------------------------------
    traj_h5 = h5py.File(args.h5_video_path, "r")
    embedding_h5 = h5py.File(args.h5_embedding_path, "r")

    training_keys = list(embedding_h5.keys())

    # Count total timesteps (same logic as metaworld_label_reward.py)
    total_timesteps = 0
    for key in training_keys:
        for traj_id in traj_h5[key].keys():
            total_timesteps += len(traj_h5[key][traj_id]["reward"])
    num_annotations = len(np.array(embedding_h5[training_keys[0]]["minilm_lang_embedding"]))
    total_timesteps = int(total_timesteps * num_annotations)

    print(f"Environments: {len(training_keys)}")
    print(f"Annotations per env: {num_annotations}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Mode: {'progress diff' if args.use_progress_diff else 'baseline'}")

    # ------------------------------------------------------------------
    # Create output h5
    # ------------------------------------------------------------------
    labeled_dataset = h5py.File(args.output_path, "w")
    labeled_dataset.create_dataset("action", (total_timesteps, 4), dtype="float32")
    labeled_dataset.create_dataset("rewards", (total_timesteps,), dtype="float32")
    labeled_dataset.create_dataset("done", (total_timesteps,), dtype="float32")
    labeled_dataset.create_dataset(
        "policy_lang_embedding", (total_timesteps, 384), dtype="float32"
    )
    labeled_dataset.create_dataset(
        "img_embedding", (total_timesteps, 768), dtype="float32"
    )
    labeled_dataset.create_dataset("env_id", (total_timesteps,), dtype="S20")

    # ------------------------------------------------------------------
    # Process trajectories
    # ------------------------------------------------------------------
    current_timestep = 0

    for key in tqdm(training_keys, desc="Environments"):
        task_text = ENVIRONMENT_TO_INSTRUCTION.get(key, key)
        lang_embeddings = np.array(embedding_h5[key]["minilm_lang_embedding"])

        for traj_id in tqdm(
            list(traj_h5[key].keys()), desc=f"  {key}", leave=False
        ):
            traj_data = traj_h5[key][traj_id]
            num_steps = len(traj_data["done"])
            save_actions = np.array(traj_data["action"])
            save_dones = np.array(traj_data["done"])

            # Raw video frames: (F, H, W, C) where F = num_steps + 1
            video_frames_raw = np.array(traj_data["img"])

            # 1) DINO embeddings for policy observation
            dino_embs = get_dino_embeddings(
                [img for img in video_frames_raw], dino_model
            )
            save_img_embeddings = dino_embs[:-1]  # (num_steps, 768)

            # 2) Robometer per-frame progress
            if args.use_server:
                progress_values = robometer_progress_per_step_server(
                    video_frames_raw, task_text, args.server_url, args.max_frames
                )
            else:
                progress_values = robometer_progress_per_step_local(
                    video_frames_raw, task_text,
                    robometer_model, robometer_tokenizer,
                    robometer_collator, robometer_exp_config,
                    args.max_frames,
                )

            # 3) Compute rewards
            if args.use_progress_diff:
                save_rewards = progress_values[1:] - progress_values[:-1]
            else:
                save_rewards = progress_values[1:]

            # 4) Save for each language annotation
            for i in range(len(lang_embeddings)):
                lang_emb = lang_embeddings[i]
                save_lang = np.tile(lang_emb, (num_steps, 1))

                end = current_timestep + num_steps
                labeled_dataset["action"][current_timestep:end] = save_actions
                labeled_dataset["rewards"][current_timestep:end] = save_rewards
                labeled_dataset["done"][current_timestep:end] = save_dones
                labeled_dataset["policy_lang_embedding"][current_timestep:end] = save_lang
                labeled_dataset["img_embedding"][current_timestep:end] = save_img_embeddings
                labeled_dataset["env_id"][current_timestep:end] = key
                current_timestep += num_steps

    labeled_dataset.close()
    traj_h5.close()
    embedding_h5.close()

    print(f"Done. Saved {current_timestep} timesteps to {args.output_path}")


if __name__ == "__main__":
    main()
