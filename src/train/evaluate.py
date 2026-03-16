import sys
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    SALICON_DIR, COCO_ANN_DIR, RUNS_DIR,
    BATCH_SIZE, DEVICE, NUM_LABELS
)
from src.seed import set_seed
from src.data.salicon_coco import SaliconCocoDataset
from src.data.transforms import build_image_transform, SaliencyTransform
from src.models.vit_multitask import build_model
from src.utils.io import load_checkpoint
from src.utils.metrics import compute_map, compute_cc, compute_sim, compute_kl_div


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()

    all_logits = []
    all_labels = []
    cc_scores = []
    sim_scores = []
    kl_scores = []

    for batch in loader:
        x = batch["image"].to(device)
        y = batch["y_cls"]
        sal_gt = batch["sal_grid"]

        logits, sal_pred = model(x)

        all_logits.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(y.numpy())

        # saliency metrics per sample
        # normalize predictions to non-negative distribution for fair comparison
        sal_pred_np = sal_pred.cpu().numpy()
        sal_gt_np = sal_gt.numpy()
        for j in range(sal_pred_np.shape[0]):
            pred_j = sal_pred_np[j].copy()
            pred_j = pred_j - pred_j.min()  # shift to non-negative
            pred_j = pred_j / (pred_j.sum() + 1e-8)  # normalize to distribution
            cc_scores.append(compute_cc(pred_j, sal_gt_np[j]))
            sim_scores.append(compute_sim(pred_j, sal_gt_np[j]))
            kl_scores.append(compute_kl_div(pred_j, sal_gt_np[j]))

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    map_score = compute_map(all_labels, all_logits)

    return {
        "mAP": map_score,
        "CC": float(np.mean(cc_scores)),
        "SIM": float(np.mean(sim_scores)),
        "KL": float(np.mean(kl_scores)),
    }


def main(mode="multitask"):
    set_seed(42)
    print(f"evaluating mode: {mode}")

    img_tfm = build_image_transform(train=False)
    sal_tfm = SaliencyTransform()

    train_ds = SaliconCocoDataset(
        img_dir=SALICON_DIR / "images" / "train",
        saliency_dir=SALICON_DIR / "train",
        coco_ann_path=COCO_ANN_DIR / "instances_train2014.json",
        split_prefix="COCO_train2014",
        img_tfm=img_tfm, sal_tfm=sal_tfm, k_labels=NUM_LABELS,
    )
    val_ds = SaliconCocoDataset(
        img_dir=SALICON_DIR / "images" / "val",
        saliency_dir=SALICON_DIR / "val",
        coco_ann_path=COCO_ANN_DIR / "instances_val2014.json",
        split_prefix="COCO_val2014",
        img_tfm=img_tfm, sal_tfm=sal_tfm, k_labels=NUM_LABELS,
        cat_to_idx=train_ds.cat_to_idx,
        cat_names=train_ds.cat_names,
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    model, _ = build_model(mode=mode, num_labels=NUM_LABELS, pretrained=False)
    ckpt_path = RUNS_DIR / f"best_{mode}.pt"
    load_checkpoint(model, ckpt_path)
    model = model.to(DEVICE)

    results = evaluate_model(model, val_loader, DEVICE)

    print(f"\n--- {mode} evaluation results ---")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    with open(RUNS_DIR / f"eval_{mode}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"results saved to runs/eval_{mode}.json")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="multitask",
                       choices=["multitask", "cls_only", "sal_only"])
    args = parser.parse_args()
    main(mode=args.mode)
