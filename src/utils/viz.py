import numpy as np
import matplotlib.pyplot as plt
import torch
from src.config import IMAGENET_MEAN, IMAGENET_STD


def denormalize_image(img_tensor):
    # undo imagenet normalization for display
    # img_tensor: (3, h, w) tensor
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    img = img.clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def show_sample(img_tensor, sal_full, sal_grid, y_cls=None, cat_names=None, save_path=None):
    # display image alongside saliency maps
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    img_np = denormalize_image(img_tensor)
    axes[0].imshow(img_np)
    axes[0].set_title("image")
    axes[0].axis("off")

    axes[1].imshow(sal_full.squeeze().cpu().numpy(), cmap="hot")
    axes[1].set_title("saliency (224x224)")
    axes[1].axis("off")

    axes[2].imshow(sal_grid.squeeze().cpu().numpy(), cmap="hot")
    axes[2].set_title("saliency (14x14)")
    axes[2].axis("off")

    if y_cls is not None and cat_names is not None:
        active = [cat_names[i] for i in range(len(cat_names)) if y_cls[i] > 0.5]
        fig.suptitle(f"labels: {', '.join(active)}", fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_explanation_comparison(img_tensor, explanations, titles, save_path=None):
    # show image + multiple explanation heatmaps side by side
    n = len(explanations) + 1
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

    img_np = denormalize_image(img_tensor)
    axes[0].imshow(img_np)
    axes[0].set_title("image")
    axes[0].axis("off")

    for i, (exp, title) in enumerate(zip(explanations, titles)):
        if isinstance(exp, torch.Tensor):
            exp = exp.cpu().numpy()
        axes[i + 1].imshow(exp.squeeze(), cmap="hot")
        axes[i + 1].set_title(title)
        axes[i + 1].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_training_curves(train_losses, val_losses, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label="train loss")
    ax.plot(val_losses, label="val loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("training curves")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_deletion_curve(fractions, scores, method_name, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fractions, scores, marker="o", markersize=3)
    ax.set_xlabel("fraction of patches removed")
    ax.set_ylabel("prediction score")
    ax.set_title(f"deletion curve - {method_name}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
