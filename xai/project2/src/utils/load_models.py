import torch
from pathlib import Path

from src.config import DEVICE, NUM_LABELS, CKPT_PATHS
from src.models.vit_multitask import build_model


def load_variant(mode, device=DEVICE):
    # mode in {"multitask", "cls_only", "sal_only"}
    ckpt_path = CKPT_PATHS[mode]
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"checkpoint missing: {ckpt_path}")

    model, _ = build_model(mode=mode, num_labels=NUM_LABELS, pretrained=False)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    print(f"loaded {mode} from {ckpt_path}, val_loss={ckpt.get('loss', '?'):.4f}")
    return model


def load_all_variants(device=DEVICE):
    return {m: load_variant(m, device=device) for m in ["multitask", "cls_only", "sal_only"]}
