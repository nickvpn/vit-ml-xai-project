import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import RUNS_DIR

FIG_DIR = RUNS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 110,
})

VARIANT_COLORS = {
    "multitask": "#1f77b4",
    "cls_only": "#ff7f0e",
    "sal_only": "#2ca02c",
}


def fig_probes():
    p = RUNS_DIR / "probes.json"
    if not p.exists():
        print("probes.json missing, skipping")
        return
    with open(p) as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    for mode, results in data.items():
        layers = results["layers"]
        axes[0].plot(layers, results["class_map"], marker="o",
                     label=mode, color=VARIANT_COLORS[mode])
        axes[1].plot(layers, results["saliency_r2"], marker="o",
                     label=mode, color=VARIANT_COLORS[mode])
        axes[2].plot(layers, results["location_acc"], marker="o",
                     label=mode, color=VARIANT_COLORS[mode])

    # plot a single random-control line averaged across variants to give a floor reference
    rand_avg = np.mean([data[m]["random_control_acc"] for m in data], axis=0)
    layers_any = data[list(data.keys())[0]]["layers"]
    axes[2].plot(layers_any, rand_avg, "--", color="gray", alpha=0.7,
                  label="random-label control (avg)")

    axes[0].set_xlabel("layer")
    axes[0].set_ylabel("class probe mAP")
    axes[0].set_title("class label decodability")
    axes[1].set_xlabel("layer")
    axes[1].set_ylabel("saliency probe $R^2$")
    axes[1].set_title("saliency decodability")
    axes[2].set_xlabel("layer")
    axes[2].set_ylabel("location probe accuracy")
    axes[2].set_title("object-location decodability")
    for a in axes:
        a.legend(fontsize=8, loc="best")
        a.grid(alpha=0.25)

    plt.tight_layout()
    out = FIG_DIR / "probes_layerwise.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


def fig_cka():
    p = RUNS_DIR / "cka.json"
    if not p.exists():
        print("cka.json missing, skipping")
        return
    with open(p) as f:
        data = json.load(f)

    pair_keys = list(data.keys())
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))

    for pair in pair_keys:
        layers = data[pair]["layers"]
        axes[0].plot(layers, data[pair]["cka_linear"], marker="o", label=f"{pair} linear")
        axes[0].plot(layers, data[pair]["cka_rbf"], marker="s", linestyle="--",
                     label=f"{pair} rbf")

    axes[0].set_xlabel("layer")
    axes[0].set_ylabel("CKA (debiased)")
    axes[0].set_title("CKA across model variants")
    axes[0].legend(fontsize=7, loc="best")
    axes[0].grid(alpha=0.25)

    for pair in pair_keys:
        layers = data[pair]["layers"]
        axes[1].plot(layers, data[pair]["procrustes"], marker="o", label=f"{pair} proc")
        axes[1].plot(layers, data[pair]["svcca"], marker="s", linestyle="--",
                     label=f"{pair} svcca")

    axes[1].set_xlabel("layer")
    axes[1].set_ylabel("similarity")
    axes[1].set_title("Procrustes and SVCCA across variants")
    axes[1].legend(fontsize=7, loc="best")
    axes[1].grid(alpha=0.25)

    plt.tight_layout()
    out = FIG_DIR / "cka_layerwise.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


def fig_clustering():
    p = RUNS_DIR / "clustering.json"
    if not p.exists():
        print("clustering.json missing, skipping")
        return
    with open(p) as f:
        data = json.load(f)

    modes = list(data.keys())
    metrics = ["category_purity", "category_nmi",
               "saliency_quartile_purity", "saliency_quartile_nmi",
               "saliency_quartile_ari", "seed_stability_ari"]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = np.arange(len(metrics))
    w = 0.25
    for i, mode in enumerate(modes):
        vals = [data[mode][m] for m in metrics]
        ax.bar(x + (i - 1) * w, vals, width=w, label=mode,
               color=VARIANT_COLORS[mode])
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=8)
    ax.set_ylabel("score")
    ax.set_title("patch-token cluster quality across variants (final layer)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    out = FIG_DIR / "clustering.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


def fig_sae():
    p = RUNS_DIR / "sae_results.json"
    if not p.exists():
        print("sae_results.json missing, skipping")
        return
    with open(p) as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))

    hist = data["history"]
    axes[0].plot(hist["epoch"], hist["recon_loss"], marker="o")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("recon MSE")
    axes[0].set_title("SAE reconstruction loss")
    axes[0].grid(alpha=0.25)

    abl = data["ablation"]
    d_cls = [a["delta_cls_mean_abs"] for a in abl]
    d_sal = [a["delta_sal_mean_abs"] for a in abl]
    axes[1].scatter(d_cls, d_sal, c="C3")
    for a in abl:
        axes[1].annotate(str(a["feature"]),
                          (a["delta_cls_mean_abs"], a["delta_sal_mean_abs"]),
                          fontsize=6, alpha=0.6)
    axes[1].set_xlabel(r"$\Delta$ classification (mean abs)")
    axes[1].set_ylabel(r"$\Delta$ saliency (mean abs)")
    axes[1].set_title("SAE feature ablation effect")
    axes[1].grid(alpha=0.25)

    plt.tight_layout()
    out = FIG_DIR / "sae_results.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


def main():
    fig_probes()
    fig_cka()
    fig_clustering()
    fig_sae()


if __name__ == "__main__":
    main()
