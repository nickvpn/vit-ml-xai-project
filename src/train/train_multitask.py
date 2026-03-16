import sys
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import src.config as cfg
from src.config import (
    SALICON_DIR, COCO_ANN_DIR, RUNS_DIR,
    BATCH_SIZE, NUM_EPOCHS, LR, WEIGHT_DECAY,
    DEVICE, NUM_LABELS
)
from src.seed import set_seed
from src.data.salicon_coco import SaliconCocoDataset
from src.data.transforms import build_image_transform, SaliencyTransform
from src.models.vit_multitask import build_model
from src.utils.io import save_checkpoint
from src.utils.metrics import compute_map


def kl_saliency_loss(pred, target):
    # pred is already a distribution (softmax output), target is also a distribution
    # KL(target || pred) = sum(target * log(target / pred))
    target_flat = target.view(target.size(0), -1)
    pred_flat = pred.view(pred.size(0), -1)
    # clamp for numerical stability
    pred_flat = pred_flat.clamp(min=1e-8)
    target_flat = target_flat.clamp(min=1e-8)
    return (target_flat * (target_flat.log() - pred_flat.log())).sum(dim=-1).mean()


def train_one_epoch(model, loader, optimizer, device, lam, mode, sal_loss_type="kl"):
    model.train()
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    running_loss = 0.0
    running_cls = 0.0
    running_sal = 0.0

    for batch in loader:
        x = batch["image"].to(device)
        y = batch["y_cls"].to(device)
        sal_tgt = batch["sal_grid"].to(device)

        optimizer.zero_grad()
        logits, sal_pred = model(x)

        # pick saliency loss
        if sal_loss_type == "kl":
            sal_loss_fn = lambda p, t: kl_saliency_loss(p, t)
        else:
            sal_loss_fn = lambda p, t: mse(p, t)

        if mode == "cls_only":
            loss = bce(logits, y)
        elif mode == "sal_only":
            loss = sal_loss_fn(sal_pred, sal_tgt)
        else:
            loss_cls = bce(logits, y)
            loss_sal = sal_loss_fn(sal_pred, sal_tgt)
            loss = loss_cls + lam * loss_sal
            running_cls += loss_cls.item()
            running_sal += loss_sal.item()

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    n = max(1, len(loader))
    return running_loss / n, running_cls / n, running_sal / n


@torch.no_grad()
def validate(model, loader, device, lam, mode, sal_loss_type="kl"):
    model.eval()
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    running_loss = 0.0
    all_logits = []
    all_labels = []

    for batch in loader:
        x = batch["image"].to(device)
        y = batch["y_cls"].to(device)
        sal_tgt = batch["sal_grid"].to(device)

        logits, sal_pred = model(x)

        if sal_loss_type == "kl":
            sal_loss_fn = lambda p, t: kl_saliency_loss(p, t)
        else:
            sal_loss_fn = lambda p, t: mse(p, t)

        if mode == "cls_only":
            loss = bce(logits, y)
        elif mode == "sal_only":
            loss = sal_loss_fn(sal_pred, sal_tgt)
        else:
            loss = bce(logits, y) + lam * sal_loss_fn(sal_pred, sal_tgt)

        running_loss += loss.item()
        all_logits.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(y.cpu().numpy())

    n = max(1, len(loader))
    avg_loss = running_loss / n

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    map_score = compute_map(all_labels, all_logits)

    return avg_loss, map_score


def main(mode="multitask"):
    set_seed(42)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"training mode: {mode}")
    print(f"device: {DEVICE}")

    # transforms
    train_img_tfm = build_image_transform(train=True)
    val_img_tfm = build_image_transform(train=False)
    sal_tfm = SaliencyTransform()

    # datasets
    train_ds = SaliconCocoDataset(
        img_dir=SALICON_DIR / "images" / "train",
        saliency_dir=SALICON_DIR / "train",
        coco_ann_path=COCO_ANN_DIR / "instances_train2014.json",
        split_prefix="COCO_train2014",
        img_tfm=train_img_tfm,
        sal_tfm=sal_tfm,
        k_labels=NUM_LABELS,
    )
    val_ds = SaliconCocoDataset(
        img_dir=SALICON_DIR / "images" / "val",
        saliency_dir=SALICON_DIR / "val",
        coco_ann_path=COCO_ANN_DIR / "instances_val2014.json",
        split_prefix="COCO_val2014",
        img_tfm=val_img_tfm,
        sal_tfm=sal_tfm,
        k_labels=NUM_LABELS,
        cat_to_idx=train_ds.cat_to_idx,
        cat_names=train_ds.cat_names,
    )

    print(f"train samples: {len(train_ds)}, val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    # model
    model, _ = build_model(mode=mode, num_labels=NUM_LABELS, pretrained=True)
    model = model.to(DEVICE)

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # lr scheduler, cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # training
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_map": []}

    sal_loss_type = cfg.SAL_LOSS_TYPE
    lam = cfg.SAL_LOSS_WEIGHT
    print(f"saliency loss: {sal_loss_type}, lambda: {lam}")

    for epoch in range(NUM_EPOCHS):
        train_loss, cls_loss, sal_loss = train_one_epoch(
            model, train_loader, optimizer, DEVICE, lam, mode, sal_loss_type
        )
        val_loss, val_map = validate(model, val_loader, DEVICE, lam, mode, sal_loss_type)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch+1}/{NUM_EPOCHS} | "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"val_mAP={val_map:.4f} | lr={lr_now:.6f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_map"].append(val_map)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss,
                          f"best_{mode}.pt")

    # save final checkpoint and history
    save_checkpoint(model, optimizer, NUM_EPOCHS - 1, val_loss,
                   f"final_{mode}.pt")

    with open(RUNS_DIR / f"history_{mode}.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"training complete. history saved to runs/history_{mode}.json")

    return model, history


def main_with_lambda(mode="multitask", lam=None, sal_loss_type=None, suffix=None):
    # convenience wrapper for lambda ablation runs
    if lam is not None:
        cfg.SAL_LOSS_WEIGHT = lam
    if sal_loss_type is not None:
        cfg.SAL_LOSS_TYPE = sal_loss_type

    # override suffix for saving if provided
    model, history = main(mode=mode)

    if suffix:
        # save with custom suffix
        with open(RUNS_DIR / f"history_{suffix}.json", "w") as f:
            json.dump(history, f, indent=2)

    return model, history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="multitask",
                       choices=["multitask", "cls_only", "sal_only"])
    parser.add_argument("--lam", type=float, default=None,
                       help="override SAL_LOSS_WEIGHT")
    parser.add_argument("--sal_loss", type=str, default=None,
                       choices=["mse", "kl"],
                       help="override SAL_LOSS_TYPE")
    args = parser.parse_args()

    if args.lam is not None:
        cfg.SAL_LOSS_WEIGHT = args.lam
    if args.sal_loss is not None:
        cfg.SAL_LOSS_TYPE = args.sal_loss

    main(mode=args.mode)
