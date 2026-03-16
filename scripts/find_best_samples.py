"""scan val set to find the most presentable images for the report.
looks for: high model confidence, focused saliency, clear single-object scenes."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np

from src.config import SALICON_DIR, COCO_ANN_DIR, NUM_LABELS, RUNS_DIR, DEVICE
from src.seed import set_seed
from src.data.salicon_coco import SaliconCocoDataset
from src.data.transforms import build_image_transform, SaliencyTransform
from src.models.vit_multitask import build_model
from src.utils.io import load_checkpoint


def saliency_focus_score(sal_grid):
    # how concentrated is the saliency? higher = more focused on a clear region
    sal = sal_grid.squeeze().numpy()
    # ratio of energy in top 25% of patches vs total
    flat = sal.flatten()
    threshold = np.percentile(flat, 75)
    top_energy = flat[flat >= threshold].sum()
    total_energy = flat.sum() + 1e-8
    return top_energy / total_energy


def main():
    set_seed(42)

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

    model, _ = build_model(mode="multitask", num_labels=NUM_LABELS, pretrained=False)
    load_checkpoint(model, RUNS_DIR / "best_multitask.pt")
    model = model.to(DEVICE)
    model.eval()

    cat_names = train_ds.cat_names

    # scan first 500 val samples
    n_scan = min(500, len(val_ds))
    candidates = []

    print(f"scanning {n_scan} val samples...")
    for idx in range(n_scan):
        sample = val_ds[idx]
        image = sample["image"].unsqueeze(0)
        y_cls = sample["y_cls"]
        sal_grid = sample["sal_grid"]

        # number of active labels (prefer 1-2 objects)
        n_labels = int(y_cls.sum().item())

        # saliency focus
        focus = saliency_focus_score(sal_grid)

        # model confidence
        with torch.no_grad():
            logits, _ = model(image.to(DEVICE))
            probs = torch.sigmoid(logits[0]).cpu()

        top_prob = probs.max().item()
        top_class = int(probs.argmax().item())

        # active category names
        active_cats = [cat_names[i] for i in range(len(cat_names)) if y_cls[i] > 0.5]

        # score: prefer single-object, high confidence, focused saliency
        score = (
            top_prob * 0.4
            + focus * 0.4
            + (1.0 if n_labels == 1 else 0.5 if n_labels == 2 else 0.2) * 0.2
        )

        candidates.append({
            "idx": idx,
            "score": score,
            "top_prob": top_prob,
            "top_class": cat_names[top_class],
            "n_labels": n_labels,
            "active_cats": active_cats,
            "focus": focus,
        })

        if idx % 100 == 0:
            print(f"  {idx}/{n_scan}")

    # sort by score
    candidates.sort(key=lambda x: x["score"], reverse=True)

    print("\n--- TOP 20 OVERALL ---")
    print(f"{'rank':<5} {'idx':<6} {'score':<7} {'prob':<7} {'focus':<7} {'#lab':<5} {'top_class':<15} {'categories'}")
    print("-" * 80)
    for rank, c in enumerate(candidates[:20]):
        print(f"{rank+1:<5} {c['idx']:<6} {c['score']:.3f}   {c['top_prob']:.3f}   {c['focus']:.3f}   "
              f"{c['n_labels']:<5} {c['top_class']:<15} {', '.join(c['active_cats'])}")

    # best per category (for diversity)
    best_per_cat = {}
    for c in candidates:
        cat = c["top_class"]
        if cat not in best_per_cat or c["score"] > best_per_cat[cat]["score"]:
            best_per_cat[cat] = c

    print("\n--- BEST PER CATEGORY (diverse selection) ---")
    print(f"{'idx':<6} {'score':<7} {'prob':<7} {'focus':<7} {'#lab':<5} {'top_class':<15} {'categories'}")
    print("-" * 80)
    for cat in sorted(best_per_cat.keys()):
        c = best_per_cat[cat]
        print(f"{c['idx']:<6} {c['score']:.3f}   {c['top_prob']:.3f}   {c['focus']:.3f}   "
              f"{c['n_labels']:<5} {c['top_class']:<15} {', '.join(c['active_cats'])}")

    # best multi-label (2 objects)
    multi = [c for c in candidates if c["n_labels"] == 2]
    multi.sort(key=lambda x: x["score"], reverse=True)
    print("\n--- TOP 10 MULTI-LABEL (2 objects, interesting for XAI) ---")
    print(f"{'idx':<6} {'score':<7} {'prob':<7} {'focus':<7} {'categories'}")
    print("-" * 60)
    for c in multi[:10]:
        print(f"{c['idx']:<6} {c['score']:.3f}   {c['top_prob']:.3f}   {c['focus']:.3f}   {', '.join(c['active_cats'])}")


if __name__ == "__main__":
    main()
