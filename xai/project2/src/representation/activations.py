import torch
import numpy as np
from torch.utils.data import DataLoader

from src.config import (
    SALICON_DIR, COCO_ANN_DIR, NUM_LABELS, GRID_SIZE,
    ANALYSIS_BATCH_SIZE, DEVICE
)
from src.data.salicon_coco import SaliconCocoDataset
from src.data.transforms import build_image_transform, SaliencyTransform


def get_block_outputs(model):
    # registers forward hooks on each transformer block to capture residual stream output
    # returns (handles, storage_dict) with storage_dict[layer_idx] -> tensor list
    storage = {}
    handles = []

    def make_hook(idx):
        def hook(_module, _inp, out):
            # out is (b, tokens, d) for ViT
            storage[idx] = out.detach()
        return hook

    for i, block in enumerate(model.vit.blocks):
        h = block.register_forward_hook(make_hook(i))
        handles.append(h)

    return handles, storage


def remove_hooks(handles):
    for h in handles:
        h.remove()


def build_val_loader(n_samples=None, batch_size=ANALYSIS_BATCH_SIZE):
    img_tfm = build_image_transform(train=False)
    sal_tfm = SaliencyTransform()
    ds = SaliconCocoDataset(
        img_dir=SALICON_DIR / "images" / "val",
        saliency_dir=SALICON_DIR / "val",
        coco_ann_path=COCO_ANN_DIR / "instances_val2014.json",
        split_prefix="COCO_val2014",
        img_tfm=img_tfm,
        sal_tfm=sal_tfm,
        k_labels=NUM_LABELS,
    )
    if n_samples is not None:
        # subset to a deterministic head slice
        from torch.utils.data import Subset
        ds = Subset(ds, list(range(min(n_samples, len(ds)))))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=True)
    return loader


@torch.no_grad()
def collect_activations(model, loader, device=DEVICE, layers=None, token="cls"):
    # token: "cls" -> shape (n, d); "patch_mean" -> mean over patch tokens (n, d);
    #        "patch_all" -> (n, num_patches, d) for clustering / sae
    handles, storage = get_block_outputs(model)
    feats_per_layer = {}
    labels_y = []
    labels_sal_full = []
    img_ids = []

    if layers is None:
        layers = list(range(len(model.vit.blocks)))

    for batch in loader:
        x = batch["image"].to(device)
        _ = model(x)
        for li in layers:
            t = storage[li]
            if token == "cls":
                v = t[:, 0, :].cpu().numpy()
            elif token == "patch_mean":
                v = t[:, 1:, :].mean(dim=1).cpu().numpy()
            elif token == "patch_all":
                v = t[:, 1:, :].cpu().numpy()
            else:
                raise ValueError(f"unknown token mode: {token}")
            feats_per_layer.setdefault(li, []).append(v)
        labels_y.append(batch["y_cls"].numpy())
        labels_sal_full.append(batch["sal_full"].numpy())
        img_ids.append(np.array(batch["img_id"]))

    remove_hooks(handles)

    out = {li: np.concatenate(feats_per_layer[li], axis=0) for li in layers}
    y = np.concatenate(labels_y, axis=0)
    sal = np.concatenate(labels_sal_full, axis=0)
    iids = np.concatenate(img_ids, axis=0)
    return out, {"y_cls": y, "sal_full": sal, "img_ids": iids}
