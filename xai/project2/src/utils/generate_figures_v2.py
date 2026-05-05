"""richer figures for the slide deck.

generates:
- cka_heatmap.png: 12-layer heatmap of procrustes/svcca for each pair
- final_layer_bars.png: focused bar chart of all four similarity metrics at layer 11
- probes_diff.png: per-layer probe-score differences with a band
- sae_taxonomy.png: zoomed sae feature ablation plane with class-leaning region shaded
- variant_radar.png: radar/bar comparison summarizing each variant on three axes
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

from src.config import RUNS_DIR

FIG_DIR = RUNS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 130,
})

VC = {
    "multitask": "#1f77b4",
    "cls_only": "#ff7f0e",
    "sal_only": "#2ca02c",
}


def fig_cka_heatmap():
    p = RUNS_DIR / "cka.json"
    with open(p) as f:
        data = json.load(f)

    pair_keys = ["multitask_vs_cls_only",
                 "multitask_vs_sal_only",
                 "cls_only_vs_sal_only"]
    metrics = ["cka_linear", "cka_rbf", "procrustes", "svcca"]
    metric_titles = ["Linear CKA", "RBF CKA", "Procrustes", "SVCCA"]

    fig, axes = plt.subplots(1, 4, figsize=(13, 2.6),
                              gridspec_kw={"wspace": 0.18})
    layers = data[pair_keys[0]]["layers"]
    n_layers = len(layers)

    for ax, metric, title in zip(axes, metrics, metric_titles):
        mat = np.array([data[k][metric] for k in pair_keys])
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_yticks(range(len(pair_keys)))
        ax.set_yticklabels(["mt vs\ncls", "mt vs\nsal", "cls vs\nsal"], fontsize=8)
        ax.set_xticks(range(n_layers))
        ax.set_xticklabels(layers, fontsize=7)
        ax.set_xlabel("layer", fontsize=9)
        ax.set_title(title, fontsize=10)
        # numeric annotations on every other layer to avoid clutter
        for i in range(len(pair_keys)):
            for j in range(n_layers):
                if j % 2 == 0:
                    val = mat[i, j]
                    color = "black" if val > 0.5 else "white"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            color=color, fontsize=6)
    cax = fig.add_axes([0.93, 0.18, 0.012, 0.66])
    plt.colorbar(im, cax=cax)
    out = FIG_DIR / "cka_heatmap.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


def fig_final_layer_bars():
    p = RUNS_DIR / "cka.json"
    with open(p) as f:
        data = json.load(f)

    pair_keys = ["multitask_vs_cls_only",
                 "multitask_vs_sal_only",
                 "cls_only_vs_sal_only"]
    pair_labels = ["multi-task ↔ cls-only", "multi-task ↔ sal-only", "cls-only ↔ sal-only"]
    metrics = ["cka_linear", "cka_rbf", "procrustes", "svcca"]
    metric_labels = ["Linear CKA", "RBF CKA", "Procrustes", "SVCCA"]

    fig, ax = plt.subplots(figsize=(8.4, 4))
    n_pairs = len(pair_keys)
    n_metrics = len(metrics)
    x = np.arange(n_pairs)
    bw = 0.20
    palette = ["#a6cee3", "#1f78b4", "#b2df8a", "#33a02c"]

    for i, (m, ml) in enumerate(zip(metrics, metric_labels)):
        vals = [data[pk][m][-1] for pk in pair_keys]  # final layer
        bars = ax.bar(x + (i - 1.5) * bw, vals, width=bw, label=ml,
                       color=palette[i], edgecolor="black", linewidth=0.4)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("similarity")
    ax.set_title("Final-layer pairwise representation similarity (4 metrics)")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = FIG_DIR / "final_layer_bars.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


def fig_probes_focused():
    p = RUNS_DIR / "probes.json"
    with open(p) as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))

    # left: class mAP & saliency R2 with shaded region between cls_only and multitask
    layers = data["multitask"]["layers"]
    for mode in ["multitask", "cls_only", "sal_only"]:
        axes[0].plot(layers, data[mode]["class_map"], marker="o",
                     color=VC[mode], label=f"{mode}", linewidth=1.6)
    axes[0].set_xlabel("layer")
    axes[0].set_ylabel("class probe mAP")
    axes[0].set_title("(a) class decodability per layer")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].grid(alpha=0.25)
    axes[0].axhline(np.mean([data[m]["random_control_acc"][-1] for m in data]),
                     linestyle="--", color="gray", alpha=0.5,
                     label="random control")

    # right: saliency R2
    for mode in ["multitask", "cls_only", "sal_only"]:
        axes[1].plot(layers, data[mode]["saliency_r2"], marker="s",
                     color=VC[mode], label=f"{mode}", linewidth=1.6)
    axes[1].set_xlabel("layer")
    axes[1].set_ylabel("saliency probe $R^2$")
    axes[1].set_title("(b) saliency decodability per layer")
    axes[1].legend(fontsize=8, loc="lower right")
    axes[1].grid(alpha=0.25)
    plt.tight_layout()
    out = FIG_DIR / "probes_focused.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


def fig_sae_taxonomy():
    p = RUNS_DIR / "sae_results.json"
    with open(p) as f:
        data = json.load(f)

    abl = data["ablation"]
    d_cls = np.array([a["delta_cls_mean_abs"] for a in abl])
    d_sal = np.array([a["delta_sal_mean_abs"] for a in abl])
    feats = [a["feature"] for a in abl]
    fracs = [a["frac_active"] for a in abl]

    fig, ax = plt.subplots(figsize=(7.8, 4.2))

    # define quadrant thresholds at the median of each axis
    cx = float(np.median(d_cls))
    cy = float(np.median(d_sal))

    # shade quadrants
    xlim = max(d_cls) * 1.12
    ylim = max(d_sal) * 1.5
    ax.axvspan(cx, xlim, ymin=0, ymax=cy/ylim, color="#fff3b0", alpha=0.45)
    ax.axhspan(cy, ylim, xmin=0, xmax=cx/xlim, color="#cdeac0", alpha=0.45)
    ax.axvspan(cx, xlim, ymin=cy/ylim, ymax=1.0, color="#f4a3a3", alpha=0.30)
    ax.axvspan(0, cx, ymin=0, ymax=cy/ylim, color="#dddddd", alpha=0.30)

    # quadrant labels
    ax.text(xlim*0.97, cy*0.45, "class-leaning", ha="right", va="center",
            fontsize=9, color="#806600")
    ax.text(cx*0.45, ylim*0.92, "saliency-leaning", ha="center", va="top",
            fontsize=9, color="#3c6e3c")
    ax.text(xlim*0.97, ylim*0.92, "shared", ha="right", va="top",
            fontsize=9, color="#a13c3c")
    ax.text(cx*0.45, cy*0.45, "low-effect", ha="center", va="center",
            fontsize=9, color="#666")

    sizes = 30 + np.array(fracs) * 350
    sc = ax.scatter(d_cls, d_sal, s=sizes, c="#3a4a8a",
                     edgecolors="black", alpha=0.75)
    # annotate top-3 by total ablation effect
    total = d_cls + 8 * d_sal  # weight saliency since axes differ
    top3 = np.argsort(-total)[:5]
    for idx in top3:
        ax.annotate(f"feat {feats[idx]}",
                     (d_cls[idx], d_sal[idx]),
                     xytext=(8, 4), textcoords="offset points",
                     fontsize=8, color="black")

    ax.set_xlabel(r"$|\Delta$ classification$|$  (mean over patches)")
    ax.set_ylabel(r"$|\Delta$ saliency$|$  (mean over patches)")
    ax.set_title("SAE feature ablation plane (multi-task, block 6)")
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, ylim)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    out = FIG_DIR / "sae_taxonomy.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


def fig_variant_summary():
    """final-layer summary of each variant on three normalized axes."""
    pp = RUNS_DIR / "probes.json"
    cl = RUNS_DIR / "clustering.json"
    with open(pp) as f:
        probes = json.load(f)
    with open(cl) as f:
        cluster = json.load(f)

    modes = ["multitask", "cls_only", "sal_only"]
    fig, ax = plt.subplots(figsize=(8.4, 3.6))

    metrics = [
        ("class probe mAP", lambda m: probes[m]["class_map"][-1]),
        ("saliency probe $R^2$", lambda m: probes[m]["saliency_r2"][-1]),
        ("location probe acc.", lambda m: probes[m]["location_acc"][-1]),
        ("cat. cluster NMI", lambda m: cluster[m]["category_nmi"]),
        ("sal. cluster NMI", lambda m: cluster[m]["saliency_quartile_nmi"]),
    ]
    n_metrics = len(metrics)
    x = np.arange(n_metrics)
    bw = 0.26
    for i, mode in enumerate(modes):
        vals = [m_fn(mode) for _, m_fn in metrics]
        bars = ax.bar(x + (i - 1) * bw, vals, width=bw, label=mode,
                       color=VC[mode], edgecolor="black", linewidth=0.4)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics], fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("score")
    ax.set_title("Final-layer probe / clustering summary by variant")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    out = FIG_DIR / "variant_summary.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


def fig_attribution_bars():
    gc = RUNS_DIR / "grad_cam_results.json"
    sh = RUNS_DIR / "shap_results.json"
    with open(gc) as f:
        g = json.load(f)["summary"]
    with open(sh) as f:
        s = json.load(f)["summary"]

    methods = ["Grad-CAM\n(class method)", "kernel SHAP\n(additional)"]
    metrics = ["mean_del_auc", "mean_ins_auc", "mean_align_cc", "mean_align_sim"]
    metric_labels = ["Del AUC ↓", "Ins AUC ↑", "Align CC ↑", "Align SIM ↑"]
    g_vals = [g[k] for k in metrics]
    s_vals = [s[k] for k in metrics]

    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    x = np.arange(len(metrics))
    bw = 0.36
    bg = ax.bar(x - bw/2, g_vals, width=bw, label=methods[0],
                color="#5b8def", edgecolor="black", linewidth=0.4)
    bs = ax.bar(x + bw/2, s_vals, width=bw, label=methods[1],
                color="#e08d8d", edgecolor="black", linewidth=0.4)
    for b, v in list(zip(bg, g_vals)) + list(zip(bs, s_vals)):
        ax.text(b.get_x()+b.get_width()/2, v + 0.015 if v >= 0 else v - 0.04,
                f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top",
                fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylabel("score")
    ax.set_title("Grad-CAM (class-method baseline) vs. SHAP on the multi-task variant")
    ax.legend(loc="upper right", fontsize=9)
    ax.axhline(0, color="gray", linewidth=0.6)
    ax.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    out = FIG_DIR / "attribution_bars.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


def main():
    fig_cka_heatmap()
    fig_final_layer_bars()
    fig_probes_focused()
    fig_sae_taxonomy()
    fig_variant_summary()
    fig_attribution_bars()


if __name__ == "__main__":
    main()
