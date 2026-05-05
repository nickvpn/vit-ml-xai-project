import torch
from pathlib import Path
from src.config import RUNS_DIR


def save_checkpoint(model, optimizer, epoch, loss, filename):
    path = RUNS_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, path)
    print(f"checkpoint saved: {path}")


def load_checkpoint(model, path, optimizer=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"loaded checkpoint from {path}, epoch {ckpt.get('epoch', '?')}")
    return ckpt
