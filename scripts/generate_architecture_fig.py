"""generate architecture diagram for the multi-task vit model."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path


def draw_architecture(save_path):
    fig, ax = plt.subplots(1, 1, figsize=(10, 3.2), dpi=200)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    ax.axis("off")

    # colors
    c_img = "#4a90d9"
    c_patch = "#5ba5e6"
    c_backbone = "#6b7b8d"
    c_cls_head = "#e67e22"
    c_sal_head = "#27ae60"
    c_token = "#8e44ad"
    c_arrow = "#333333"

    box_kw = dict(boxstyle="round,pad=0.15")

    # --- input image ---
    # draw a small grid to represent an image
    img_x, img_y = 0.4, 1.15
    img_s = 0.9
    ax.add_patch(FancyBboxPatch((img_x, img_y), img_s, img_s,
                                facecolor=c_img, edgecolor="white",
                                linewidth=1.5, **box_kw, alpha=0.85))
    # grid lines to suggest patches
    for i in range(1, 4):
        frac = i / 4
        ax.plot([img_x + frac * img_s, img_x + frac * img_s],
                [img_y + 0.08, img_y + img_s - 0.08],
                color="white", lw=0.7, alpha=0.6)
        ax.plot([img_x + 0.08, img_x + img_s - 0.08],
                [img_y + frac * img_s, img_y + frac * img_s],
                color="white", lw=0.7, alpha=0.6)
    ax.text(img_x + img_s / 2, img_y - 0.22, "Input\n224×224",
            ha="center", va="top", fontsize=7, color="#333")

    # --- patch embedding ---
    pe_x, pe_y = 2.0, 1.05
    pe_w, pe_h = 1.1, 1.1
    ax.add_patch(FancyBboxPatch((pe_x, pe_y), pe_w, pe_h,
                                facecolor=c_patch, edgecolor="white",
                                linewidth=1.5, **box_kw, alpha=0.8))
    ax.text(pe_x + pe_w / 2, pe_y + pe_h / 2 + 0.12, "Patch",
            ha="center", va="center", fontsize=8, fontweight="bold", color="white")
    ax.text(pe_x + pe_w / 2, pe_y + pe_h / 2 - 0.12, "Embed",
            ha="center", va="center", fontsize=8, fontweight="bold", color="white")
    ax.text(pe_x + pe_w / 2, pe_y - 0.22, "16×16 patches\n→ 196 tokens",
            ha="center", va="top", fontsize=6.5, color="#555")

    # arrow: image -> patch embed
    ax.annotate("", xy=(pe_x - 0.05, 1.6), xytext=(img_x + img_s + 0.05, 1.6),
                arrowprops=dict(arrowstyle="-|>", color=c_arrow, lw=1.5))

    # --- transformer backbone ---
    bb_x, bb_y = 3.7, 0.75
    bb_w, bb_h = 1.6, 1.7
    ax.add_patch(FancyBboxPatch((bb_x, bb_y), bb_w, bb_h,
                                facecolor=c_backbone, edgecolor="white",
                                linewidth=1.5, **box_kw, alpha=0.85))
    ax.text(bb_x + bb_w / 2, bb_y + bb_h / 2 + 0.25, "DeiT-Small",
            ha="center", va="center", fontsize=9, fontweight="bold", color="white")
    ax.text(bb_x + bb_w / 2, bb_y + bb_h / 2 - 0.05, "12 Layers",
            ha="center", va="center", fontsize=7.5, color="#ddd")
    ax.text(bb_x + bb_w / 2, bb_y + bb_h / 2 - 0.3, "384-dim",
            ha="center", va="center", fontsize=7.5, color="#ddd")
    ax.text(bb_x + bb_w / 2, bb_y - 0.15, "Shared Backbone",
            ha="center", va="top", fontsize=7, color="#555", style="italic")

    # arrow: patch embed -> backbone
    ax.annotate("", xy=(bb_x - 0.05, 1.6), xytext=(pe_x + pe_w + 0.05, 1.6),
                arrowprops=dict(arrowstyle="-|>", color=c_arrow, lw=1.5))

    # --- CLS token output ---
    cls_x, cls_y = 5.9, 2.0
    cls_w, cls_h = 0.9, 0.55
    ax.add_patch(FancyBboxPatch((cls_x, cls_y), cls_w, cls_h,
                                facecolor=c_token, edgecolor="white",
                                linewidth=1.2, **box_kw, alpha=0.8))
    ax.text(cls_x + cls_w / 2, cls_y + cls_h / 2, "[CLS]",
            ha="center", va="center", fontsize=7.5, fontweight="bold", color="white")

    # --- patch tokens output ---
    pt_x, pt_y = 5.9, 0.85
    pt_w, pt_h = 0.9, 0.55
    ax.add_patch(FancyBboxPatch((pt_x, pt_y), pt_w, pt_h,
                                facecolor=c_token, edgecolor="white",
                                linewidth=1.2, **box_kw, alpha=0.8))
    ax.text(pt_x + pt_w / 2, pt_y + pt_h / 2, "Patches",
            ha="center", va="center", fontsize=7.5, fontweight="bold", color="white")
    ax.text(pt_x + pt_w / 2, pt_y - 0.15, "196 tokens",
            ha="center", va="top", fontsize=6.5, color="#555")

    # arrows: backbone -> cls and patches
    ax.annotate("", xy=(cls_x - 0.05, cls_y + cls_h / 2),
                xytext=(bb_x + bb_w + 0.05, 2.0),
                arrowprops=dict(arrowstyle="-|>", color=c_arrow, lw=1.3))
    ax.annotate("", xy=(pt_x - 0.05, pt_y + pt_h / 2),
                xytext=(bb_x + bb_w + 0.05, 1.2),
                arrowprops=dict(arrowstyle="-|>", color=c_arrow, lw=1.3))

    # --- classification head ---
    ch_x, ch_y = 7.4, 1.95
    ch_w, ch_h = 1.3, 0.65
    ax.add_patch(FancyBboxPatch((ch_x, ch_y), ch_w, ch_h,
                                facecolor=c_cls_head, edgecolor="white",
                                linewidth=1.5, **box_kw, alpha=0.85))
    ax.text(ch_x + ch_w / 2, ch_y + ch_h / 2 + 0.08, "Linear",
            ha="center", va="center", fontsize=8, fontweight="bold", color="white")
    ax.text(ch_x + ch_w / 2, ch_y + ch_h / 2 - 0.14, "384 → 20",
            ha="center", va="center", fontsize=7, color="#fff", alpha=0.9)

    # arrow: cls -> cls head
    ax.annotate("", xy=(ch_x - 0.05, ch_y + ch_h / 2),
                xytext=(cls_x + cls_w + 0.05, cls_y + cls_h / 2),
                arrowprops=dict(arrowstyle="-|>", color=c_arrow, lw=1.3))

    # --- saliency head ---
    sh_x, sh_y = 7.4, 0.75
    sh_w, sh_h = 1.3, 0.65
    ax.add_patch(FancyBboxPatch((sh_x, sh_y), sh_w, sh_h,
                                facecolor=c_sal_head, edgecolor="white",
                                linewidth=1.5, **box_kw, alpha=0.85))
    ax.text(sh_x + sh_w / 2, sh_y + sh_h / 2 + 0.08, "Linear+Softmax",
            ha="center", va="center", fontsize=7.5, fontweight="bold", color="white")
    ax.text(sh_x + sh_w / 2, sh_y + sh_h / 2 - 0.14, "384 → 1",
            ha="center", va="center", fontsize=7, color="#fff", alpha=0.9)

    # arrow: patches -> sal head
    ax.annotate("", xy=(sh_x - 0.05, sh_y + sh_h / 2),
                xytext=(pt_x + pt_w + 0.05, pt_y + pt_h / 2),
                arrowprops=dict(arrowstyle="-|>", color=c_arrow, lw=1.3))

    # --- outputs ---
    # classification output
    out_cls_x = 9.1
    ax.text(out_cls_x, ch_y + ch_h / 2, "→  K=20 logits\n     (BCELoss)",
            ha="left", va="center", fontsize=7, color=c_cls_head, fontweight="bold")

    # saliency output
    ax.text(out_cls_x, sh_y + sh_h / 2, "→  14×14 map\n     (KL Loss)",
            ha="left", va="center", fontsize=7, color=c_sal_head, fontweight="bold")

    plt.tight_layout(pad=0.3)
    fig.savefig(save_path, bbox_inches="tight", facecolor="white", pad_inches=0.1)
    plt.close(fig)
    print(f"saved: {save_path}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent

    # save to both report figure dirs
    for d in ["report/intro_to_ml/figs", "report/xai/figs"]:
        out = root / d / "architecture.png"
        draw_architecture(out)
