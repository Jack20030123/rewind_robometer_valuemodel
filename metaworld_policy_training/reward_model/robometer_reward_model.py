"""
Robometer reward model for ReWiND policy training.

Supports two modes:
  - Direct inference: loads Robometer model locally (requires robometer package)
  - HTTP server inference: calls a running Robometer eval server (no extra deps)

Usage in config:
  reward=robometer                              # direct mode
  reward=robometer reward_model.use_server=true # server mode
"""

import numpy as np
import torch
from typing import List, Union

from reward_model.base_reward_model import BaseRewardModel
from reward_model.reward_utils import mean_pooling


class RobometerRewardModel(BaseRewardModel):
    def __init__(
        self,
        model_path: str = "aliangdw/Robometer-4B",
        device: str = "cuda",
        batch_size: int = 1,
        success_bonus: float = 64.0,
        reward_at_every_step: bool = True,
        max_frames: int = 4,
        use_server: bool = False,
        server_url: str = "http://localhost:8000",
    ):
        super().__init__(
            device=device,
            batch_size=batch_size,
            success_bonus=success_bonus,
        )
        self.reward_at_every_step = reward_at_every_step
        self.max_frames = max_frames
        self.use_server = use_server
        self.server_url = server_url.rstrip("/")
        self.model_path = model_path

        # Frame buffer: stores raw frames (H, W, C) uint8
        self._frame_buffer: List[np.ndarray] = []
        # Task text: stored when encode_text is called
        self._task_text: str = ""

        if use_server:
            print(f"[RobometerRewardModel] Using HTTP server at {self.server_url}")
        else:
            print(f"[RobometerRewardModel] Loading model from {model_path} ...")
            self._load_robometer(model_path, device)
            print("[RobometerRewardModel] Model loaded successfully")

    def _load_robometer(self, model_path: str, device: str):
        """Load Robometer model and setup inference pipeline."""
        from robometer.utils.save import load_model_from_hf
        from robometer.utils.setup_utils import setup_batch_collator

        exp_config, tokenizer, processor, model = load_model_from_hf(
            model_path=model_path,
            device=self.device,  # use self.device from BaseRewardModel
        )
        model.eval()

        self._robometer_model = model
        self._tokenizer = tokenizer
        self._processor = processor
        self._exp_config = exp_config
        self._batch_collator = setup_batch_collator(
            processor, tokenizer, exp_config, is_eval=True
        )

        # Determine discrete mode settings
        loss_config = getattr(exp_config, "loss", None)
        self._is_discrete = (
            getattr(loss_config, "progress_loss_type", "l2").lower() == "discrete"
            if loss_config
            else False
        )
        self._num_bins = (
            getattr(loss_config, "progress_discrete_bins", None)
            or getattr(exp_config.model, "progress_discrete_bins", 10)
        )

    # ------------------------------------------------------------------
    # _encode_text_batch: required by BaseRewardModel.encode_text()
    # ------------------------------------------------------------------
    def _encode_text_batch(self, text: List[str]) -> np.ndarray:
        """Encode text using MiniLM (inherited from BaseRewardModel) for wrapper compatibility."""
        with torch.no_grad():
            encoded_input = self.tokenizer(
                text, padding=False, truncation=True, return_tensors="pt"
            )
            model_output = self.model(**encoded_input)
            text_embeddings = (
                mean_pooling(model_output, encoded_input["attention_mask"])
                .cpu()
                .numpy()
            )
        return text_embeddings

    # ------------------------------------------------------------------
    # Override encode_text: store raw text + return MiniLM embedding
    # ------------------------------------------------------------------
    def encode_text(self, text: Union[str, List]) -> np.ndarray:
        if isinstance(text, list):
            self._task_text = text[0]
        else:
            self._task_text = text
        # Return MiniLM embedding for wrapper compatibility
        return super().encode_text(text)

    # ------------------------------------------------------------------
    # Override encode_images: store raw frame, return dummy embedding
    # ------------------------------------------------------------------
    def encode_images(self, images: np.ndarray) -> np.ndarray:
        """
        Store raw frame in buffer and return a dummy 2-dim embedding.

        The wrapper calls .squeeze() on the result, so (1, 1, 2) → (2,).
        This matches img_output_dim=2 and the observation space shape.

        Args:
            images: shape (1, 1, H, W, C) — single frame from MetaworldImageEmbeddingWrapper
                    Note: base class would transpose HWC→CHW, but we override entirely.
        """
        assert len(images.shape) == 5
        frame = images[0, 0]  # (H, W, C)

        # Handle both HWC and CHW input formats
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        if frame.ndim == 3 and frame.shape[0] in (1, 3) and frame.shape[-1] not in (1, 3):
            frame = np.transpose(frame, (1, 2, 0))  # CHW -> HWC
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        self._frame_buffer.append(frame.copy())

        # Return shape (1, 1, 2): after .squeeze() in wrapper → shape (2,)
        # img_output_dim=2 so observation space Box(shape=(2,)) matches
        # (using dim=2 avoids squeeze collapsing all size-1 dims to scalar)
        return np.zeros((1, 1, 2), dtype=np.float32)

    # ------------------------------------------------------------------
    # Override calculate_rewards: use buffered frames + task text
    # ------------------------------------------------------------------
    def calculate_rewards(
        self,
        encoded_texts: Union[np.ndarray, torch.Tensor],
        encoded_videos: Union[np.ndarray, torch.Tensor],
        *args,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute progress score using Robometer.

        The encoded_texts and encoded_videos from the wrapper are dummy values.
        We use self._frame_buffer and self._task_text instead.
        """
        # Determine how many frames the wrapper currently has
        if isinstance(encoded_videos, torch.Tensor):
            T = encoded_videos.shape[1]
        else:
            T = encoded_videos.shape[1] if encoded_videos.ndim >= 2 else len(self._frame_buffer)

        # Sync buffer: keep only the last T frames (handles episode reset)
        if len(self._frame_buffer) > T:
            self._frame_buffer = self._frame_buffer[-T:]

        if len(self._frame_buffer) == 0:
            return np.array([0.0])

        frames = np.stack(self._frame_buffer, axis=0)  # (T, H, W, C) uint8

        if self.use_server:
            progress = self._infer_server(frames, self._task_text)
        else:
            progress = self._infer_local(frames, self._task_text)

        return np.array([progress])

    # ------------------------------------------------------------------
    # Direct inference (requires robometer package)
    # ------------------------------------------------------------------
    def _infer_local(self, frames: np.ndarray, task: str) -> float:
        """
        Run Robometer inference locally.

        Args:
            frames: (T, H, W, C) uint8 numpy array
            task: task description string

        Returns:
            Progress score (float, 0-1)
        """
        from robometer.data.dataset_types import ProgressSample, Trajectory
        from robometer.evals.eval_server import compute_batch_outputs

        T = frames.shape[0]
        # Subsample frames to max_frames (consistent with _infer_server)
        if T > self.max_frames:
            indices = np.linspace(0, T - 1, self.max_frames, dtype=int)
            frames = frames[indices]
            T = self.max_frames
        traj = Trajectory(
            frames=frames,
            frames_shape=tuple(frames.shape),
            task=task,
            id="0",
            metadata={"subsequence_length": T},
            video_embeddings=None,
        )
        progress_sample = ProgressSample(trajectory=traj, sample_type="progress")
        batch = self._batch_collator([progress_sample])

        progress_inputs = batch["progress_inputs"]
        for key, value in progress_inputs.items():
            if hasattr(value, "to"):
                progress_inputs[key] = value.to(self.device)

        with torch.no_grad():
            results = compute_batch_outputs(
                self._robometer_model,
                self._tokenizer,
                progress_inputs,
                sample_type="progress",
                is_discrete_mode=self._is_discrete,
                num_bins=self._num_bins,
            )

        progress_pred = results.get("progress_pred", [])
        if progress_pred and len(progress_pred) > 0:
            # Return the last frame's progress score
            return float(progress_pred[0][-1])
        return 0.0

    # ------------------------------------------------------------------
    # HTTP server inference (no robometer dependency needed)
    # ------------------------------------------------------------------
    def _infer_server(self, frames: np.ndarray, task: str) -> float:
        """
        Call Robometer eval server via HTTP.

        Args:
            frames: (T, H, W, C) uint8 numpy array
            task: task description string

        Returns:
            Progress score (float, 0-1)
        """
        import requests
        import io
        import base64

        # Subsample frames to max_frames
        T = frames.shape[0]
        if T > self.max_frames:
            indices = np.linspace(0, T - 1, self.max_frames, dtype=int)
            frames = frames[indices]

        # Encode frames as base64 numpy array
        buf = io.BytesIO()
        np.save(buf, frames)
        frames_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        payload = {
            "frames_b64": frames_b64,
            "task": task,
            "sample_type": "progress",
        }

        try:
            resp = requests.post(
                f"{self.server_url}/predict",
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json()
            progress = result.get("progress", result.get("reward", 0.0))
            if isinstance(progress, list):
                return float(progress[-1])
            return float(progress)
        except Exception as e:
            print(f"[RobometerRewardModel] Server inference failed: {e}")
            return 0.0

    # ------------------------------------------------------------------
    # Abstract method stubs (not used, but required by BaseRewardModel)
    # ------------------------------------------------------------------
    def _encode_image_batch(self, images: torch.Tensor) -> np.ndarray:
        """Not used — encode_images is overridden directly."""
        return np.zeros((images.shape[0], images.shape[1], 2), dtype=np.float32)

    def _calculate_reward_batch(
        self, encoded_texts: np.ndarray, encoded_videos: np.ndarray
    ) -> np.ndarray:
        """Not used — calculate_rewards is overridden directly."""
        return np.array([0.0])

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def img_output_dim(self) -> int:
        # Dummy dimension (raw frames bypass the embedding pipeline)
        # Use 2 instead of 1 to avoid .squeeze() collapsing to scalar
        return 2

    @property
    def text_output_dim(self) -> int:
        return 384  # MiniLM compatibility

    @property
    def name(self) -> str:
        return "robometer"
