"""
Robometer HTTP inference server.

Loads the Robometer-4B model and serves progress predictions via FastAPI.
Used by RobometerRewardModel in server mode (use_server=true).
"""

import io
import base64
import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel

from robometer.utils.save import load_model_from_hf
from robometer.utils.setup_utils import setup_batch_collator
from robometer.data.dataset_types import ProgressSample, Trajectory
from robometer.evals.eval_server import compute_batch_outputs

app = FastAPI()

# Global model state
MODEL = None
TOKENIZER = None
BATCH_COLLATOR = None
EXP_CONFIG = None
DEVICE = None
IS_DISCRETE = False
NUM_BINS = 10


class PredictRequest(BaseModel):
    frames_b64: str
    task: str
    sample_type: str = "progress"


@app.on_event("startup")
def load_model():
    global MODEL, TOKENIZER, BATCH_COLLATOR, EXP_CONFIG, DEVICE, IS_DISCRETE, NUM_BINS

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[RobometerServer] Loading model on {DEVICE}...")

    exp_config, tokenizer, processor, model = load_model_from_hf(
        model_path="aliangdw/Robometer-4B",
        device=DEVICE,
    )
    model.eval()

    MODEL = model
    TOKENIZER = tokenizer
    EXP_CONFIG = exp_config
    BATCH_COLLATOR = setup_batch_collator(processor, tokenizer, exp_config, is_eval=True)

    loss_config = getattr(exp_config, "loss", None)
    IS_DISCRETE = (
        getattr(loss_config, "progress_loss_type", "l2").lower() == "discrete"
        if loss_config
        else False
    )
    NUM_BINS = (
        getattr(loss_config, "progress_discrete_bins", None)
        or getattr(exp_config.model, "progress_discrete_bins", 10)
    )

    print(f"[RobometerServer] Model loaded. discrete={IS_DISCRETE}, bins={NUM_BINS}")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.post("/predict")
def predict(req: PredictRequest):
    # Decode frames
    buf = io.BytesIO(base64.b64decode(req.frames_b64))
    frames = np.load(buf)  # (T, H, W, C) uint8

    T = frames.shape[0]
    traj = Trajectory(
        frames=frames,
        frames_shape=tuple(frames.shape),
        task=req.task,
        id="server",
        metadata={"subsequence_length": T},
        video_embeddings=None,
    )
    sample = ProgressSample(trajectory=traj, sample_type="progress")
    batch = BATCH_COLLATOR([sample])

    progress_inputs = batch["progress_inputs"]
    for key, value in progress_inputs.items():
        if hasattr(value, "to"):
            progress_inputs[key] = value.to(DEVICE)

    with torch.no_grad():
        results = compute_batch_outputs(
            MODEL,
            TOKENIZER,
            progress_inputs,
            sample_type="progress",
            is_discrete_mode=IS_DISCRETE,
            num_bins=NUM_BINS,
        )

    preds = results.get("progress_pred", [])
    if preds and len(preds) > 0:
        progress = [float(p) for p in preds[0]]
    else:
        progress = [0.0]

    return {"progress": progress}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
