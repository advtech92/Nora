"""
utils.py

Common utilities: logging setup, checkpoint saving & loading, device checks, etc.
"""

import os
import logging
import torch


def setup_logging(log_file: str = None):
    """
    Set up logging to stdout (and optionally to a file).
    """
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        root.addHandler(fh)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    checkpoint_dir: str,
    tokenizer=None,
):
    """
    Save model state, optimizer state, and tokenizer (optional) to a checkpoint file.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"nora_step_{step}.pt")
    state = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if tokenizer:
        # tokenizer.stoi is JSON‐serializable
        state["tokenizer_stoi"] = tokenizer.stoi

    torch.save(state, ckpt_path)
    logging.info(f"[checkpoint] Saved checkpoint to {ckpt_path}")


def load_checkpoint(
    ckpt_path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None
):
    """
    Load model & optimizer state from a checkpoint. Returns step.
    If optimizer is None, only loads model weights.
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    step = state.get("step", 0)
    if optimizer and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    logging.info(f"[checkpoint] Loaded checkpoint from {ckpt_path} (step {step})")
    return step


def get_default_device():
    """
    Return 'cuda' if available; otherwise 'cpu'.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"
