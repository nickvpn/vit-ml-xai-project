"""generate xai comparison figures for specific val set indices.
picks a diverse set of presentable images for the report."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.config import SALICON_DIR, COCO_ANN_DIR, NUM_LABELS, RUNS_DIR, DEVICE, GRID_SIZE
from src.seed import set_seed
from src.data.salicon_coco import SaliconCocoDataset
from src.data.transforms import build_image_transform, SaliencyTransform
from src.models.vit_multitask import build_model
from src.utils.io import load_checkpoint
from src.utils.viz import plot_explanation_comparison
from src.xai.run_xai import run_explanations_single, get_top_class


# diverse selection: person, dog, car, tv, dining table, multi-label (truck+dog, person+car)
REPORT_INDICES = [305, 323, 445, 299, 37, 268, 236, 10, 135, 335]


def main():
    set_seed(42)
    figures_dir = RUNS_DIR / "figures"
    figures_dir.mkdir(exist_ok=True)

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

    method_names = ["gradient_saliency", "grad_x_input", "integrated_gradients",
                    "attention_rollout", "lime"]

    for i, idx in enumerate(REPORT_INDICES):
        sample = val_ds[idx]
        image = sample["image"].unsqueeze(0)
        y_cls = sample["y_cls"]

        target_class = get_top_class(model, image, DEVICE)
        active_cats = [train_ds.cat_names[j] for j in range(len(train_ds.cat_names)) if y_cls[j] > 0.5]

        print(f"[{i+1}/{len(REPORT_INDICES)}] idx={idx}, predicted={train_ds.cat_names[target_class]}, "
              f"labels={active_cats}")

        explanations = run_explanations_single(model, image, target_class, DEVICE)

        save_path = figures_dir / f"report_comparison_{idx}.png"
        plot_explanation_comparison(
            sample["image"],
            [explanations[m] for m in method_names],
            method_names,
            save_path=save_path,
        )
        print(f"  saved {save_path.name}")


if __name__ == "__main__":
    main()
