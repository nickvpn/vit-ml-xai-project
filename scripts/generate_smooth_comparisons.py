"""generate xai comparison figures with bilinear interpolation for smoother heatmaps.
same as generate_report_comparisons.py but upscales 14x14 maps smoothly."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from src.config import SALICON_DIR, COCO_ANN_DIR, NUM_LABELS, RUNS_DIR, DEVICE, GRID_SIZE
from src.seed import set_seed
from src.data.salicon_coco import SaliconCocoDataset
from src.data.transforms import build_image_transform, SaliencyTransform
from src.models.vit_multitask import build_model
from src.utils.io import load_checkpoint
from src.utils.viz import denormalize_image
from src.xai.run_xai import run_explanations_single, get_top_class


REPORT_INDICES = [305, 323, 445, 299, 37, 268, 236, 10, 135, 335]


def smooth_heatmap(heatmap, target_size=224):
    """upscale a 14x14 heatmap to target_size using bilinear (order=1) interpolation."""
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.cpu().numpy()
    h = heatmap.squeeze()
    scale = target_size / h.shape[0]
    return zoom(h, scale, order=1)


def plot_smooth_comparison(img_tensor, explanations, titles, save_path=None):
    """same layout as plot_explanation_comparison but with smooth heatmaps."""
    n = len(explanations) + 1
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

    img_np = denormalize_image(img_tensor)
    axes[0].imshow(img_np)
    axes[0].set_title("image")
    axes[0].axis("off")

    for i, (exp, title) in enumerate(zip(explanations, titles)):
        smooth = smooth_heatmap(exp, target_size=224)
        axes[i + 1].imshow(smooth, cmap="hot", interpolation="bilinear")
        axes[i + 1].set_title(title)
        axes[i + 1].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def main():
    set_seed(42)
    figures_dir = RUNS_DIR / "figures" / "smooth"
    figures_dir.mkdir(parents=True, exist_ok=True)

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

        save_path = figures_dir / f"smooth_comparison_{idx}.png"
        plot_smooth_comparison(
            sample["image"],
            [explanations[m] for m in method_names],
            method_names,
            save_path=save_path,
        )
        print(f"  saved {save_path.name}")


if __name__ == "__main__":
    main()
