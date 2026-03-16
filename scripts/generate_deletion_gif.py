"""generate an animated GIF showing the deletion test for a sample image.
side-by-side: gradient saliency vs LIME deletion, with confidence score overlay."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image as PILImage

from src.config import SALICON_DIR, COCO_ANN_DIR, NUM_LABELS, RUNS_DIR, DEVICE, GRID_SIZE, IMG_SIZE
from src.seed import set_seed
from src.data.salicon_coco import SaliconCocoDataset
from src.data.transforms import build_image_transform, SaliencyTransform
from src.models.vit_multitask import build_model
from src.utils.io import load_checkpoint
from src.utils.viz import denormalize_image
from src.xai.run_xai import run_explanations_single, get_top_class
from src.xai.human_alignment import resize_to_grid


SAMPLE_IDX = 323  # dog + hydrant
N_STEPS = 20  # more steps = smoother animation


def build_masked_image(image, ranked_indices, n_to_mask, device):
    # mask the top n_to_mask patches on the image
    patch_h = IMG_SIZE // GRID_SIZE
    patch_w = IMG_SIZE // GRID_SIZE

    masked = image.clone()
    for idx in ranked_indices[:n_to_mask]:
        row = idx // GRID_SIZE
        col = idx % GRID_SIZE
        masked[0, :,
               row * patch_h:(row + 1) * patch_h,
               col * patch_w:(col + 1) * patch_w] = 0.0
    return masked


def get_score(model, image, target_class, device):
    with torch.no_grad():
        logits, _ = model(image.to(device))
        return torch.sigmoid(logits[0, target_class]).item()


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

    sample = val_ds[SAMPLE_IDX]
    image = sample["image"].unsqueeze(0)
    target_class = get_top_class(model, image, DEVICE)
    class_name = train_ds.cat_names[target_class]
    print(f"sample {SAMPLE_IDX}, class: {class_name}")

    # generate explanations
    print("computing explanations...")
    explanations = run_explanations_single(model, image, target_class, DEVICE)

    methods = {
        "Gradient Saliency": explanations["gradient_saliency"],
        "LIME": explanations["lime"],
    }

    n_patches = GRID_SIZE * GRID_SIZE
    frames = []

    # pre-compute rankings for each method
    rankings = {}
    for name, attr in methods.items():
        flat = attr.flatten()
        rankings[name] = np.argsort(flat)[::-1]

    # pre-compute all scores for the curve
    all_scores = {name: [] for name in methods}
    all_fracs = []

    for step in range(N_STEPS + 1):
        frac = step / N_STEPS
        n_to_mask = int(frac * n_patches)
        all_fracs.append(frac)

        for name in methods:
            masked = build_masked_image(image, rankings[name], n_to_mask, DEVICE)
            score = get_score(model, masked, target_class, DEVICE)
            all_scores[name].append(score)

    print("generating frames...")
    for step in range(N_STEPS + 1):
        frac = step / N_STEPS
        n_to_mask = int(frac * n_patches)

        fig = plt.figure(figsize=(14, 5))
        gs = GridSpec(1, 3, width_ratios=[1, 1, 1.2], figure=fig)

        method_names = list(methods.keys())

        for i, name in enumerate(method_names):
            ax = fig.add_subplot(gs[0, i])
            masked = build_masked_image(image, rankings[name], n_to_mask, DEVICE)
            masked_np = denormalize_image(masked.squeeze(0))
            ax.imshow(masked_np)
            score = all_scores[name][step]
            ax.set_title(f"{name}\nconf: {score:.3f}", fontsize=11)
            ax.axis("off")

        # deletion curve panel
        ax_curve = fig.add_subplot(gs[0, 2])
        colors = {"Gradient Saliency": "#e15759", "LIME": "#4e79a7"}
        for name in method_names:
            # plot full curve faded, current progress solid
            ax_curve.plot(all_fracs, all_scores[name],
                         color=colors[name], alpha=0.25, linewidth=1)
            ax_curve.plot(all_fracs[:step+1], all_scores[name][:step+1],
                         color=colors[name], linewidth=2.5, label=name)
            # current point
            ax_curve.scatter([all_fracs[step]], [all_scores[name][step]],
                           color=colors[name], s=60, zorder=5)

        ax_curve.set_xlabel("fraction of patches removed", fontsize=10)
        ax_curve.set_ylabel("prediction confidence", fontsize=10)
        ax_curve.set_title("Deletion Test", fontsize=11)
        ax_curve.set_xlim(-0.02, 1.02)
        ax_curve.set_ylim(-0.05, 1.05)
        ax_curve.legend(loc="upper right", fontsize=9)
        ax_curve.grid(True, alpha=0.3)

        fig.suptitle(f'Faithfulness: removing important patches ({class_name})',
                    fontsize=13, y=0.98)
        plt.tight_layout()

        # save frame to buffer
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        frame_data = np.asarray(buf)
        # convert RGBA to RGB
        frames.append(PILImage.fromarray(frame_data[:, :, :3]))
        plt.close(fig)

        if step % 5 == 0:
            print(f"  frame {step}/{N_STEPS}")

    # save GIF
    gif_path = figures_dir / "deletion_test_animation.gif"
    # hold first and last frames longer
    durations = [800] + [250] * (len(frames) - 2) + [1500]
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
    )
    print(f"saved {gif_path}")


if __name__ == "__main__":
    main()
