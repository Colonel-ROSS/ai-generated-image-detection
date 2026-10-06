"""
utils/checkpoint.py

Checkpoint manager for saving and resuming training state.
Called by all training scripts at the end of every epoch and subtask.
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_JSON = PROJECT_ROOT / "progress" / "CHECKPOINT.json"
LOGS_DIR = PROJECT_ROOT / "progress" / "logs"
MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints"


def load_checkpoint_json() -> dict:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if CHECKPOINT_JSON.exists():
        with open(CHECKPOINT_JSON) as f:
            return json.load(f)
    return {}


def save_checkpoint_json(state: dict):
    state["last_updated"] = datetime.utcnow().isoformat()
    CHECKPOINT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_JSON, "w") as f:
        json.dump(state, f, indent=2)

    # Also write a timestamped log entry
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"log_{ts}.json"
    with open(log_path, "w") as f:
        json.dump(state, f, indent=2)


def save_model_checkpoint(
    model: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    val_acc: float,
    task_name: str,
    extra: dict = None,
):
    """Save a model checkpoint and update CHECKPOINT.json."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{task_name}_epoch{epoch:03d}_{ts}.pth"
    filepath = MODELS_DIR / filename

    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimiser_state_dict": optimiser.state_dict(),
        "val_loss": val_loss,
        "val_acc": val_acc,
        "task_name": task_name,
        "timestamp": ts,
    }
    if extra:
        payload.update(extra)

    torch.save(payload, filepath)

    # Update CHECKPOINT.json
    state = load_checkpoint_json()
    if "models_saved" not in state:
        state["models_saved"] = []
    state["models_saved"].append(str(filepath))
    state["last_model"] = str(filepath)
    state["last_val_loss"] = val_loss
    state["last_val_acc"] = val_acc
    state["last_epoch"] = epoch
    save_checkpoint_json(state)

    print(f"[Checkpoint] Saved model: {filename}")
    return filepath


def load_model_checkpoint(filepath: str, model: torch.nn.Module, optimiser=None, device="cpu"):
    """Load a model checkpoint."""
    payload = torch.load(filepath, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    if optimiser is not None and "optimiser_state_dict" in payload:
        optimiser.load_state_dict(payload["optimiser_state_dict"])
    print(f"[Checkpoint] Loaded: {filepath} (epoch {payload.get('epoch', '?')})")
    return payload


def save_resume_note(note: str):
    """Write a human-readable resume note for when context runs out."""
    path = PROJECT_ROOT / "progress" / "resume_note.txt"
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    with open(path, "w") as f:
        f.write(f"Last updated: {ts}\n\n")
        f.write(note)
    print(f"[Checkpoint] Resume note written to {path}")


def update_task_status(task_id: str, status: str, notes: str = ""):
    """Update a task's status in CHECKPOINT.json."""
    state = load_checkpoint_json()
    if "tasks" not in state:
        state["tasks"] = {}
    state["tasks"][task_id] = {
        "status": status,
        "notes": notes,
        "updated": datetime.utcnow().isoformat(),
    }
    state["current_task"] = task_id
    save_checkpoint_json(state)
    print(f"[Checkpoint] Task {task_id} -> {status}")
