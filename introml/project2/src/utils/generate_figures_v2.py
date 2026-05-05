"""richer figures for the introml slide deck.

generates:
- regime_annotated.png: double-descent curve with shaded under/threshold/over regions
- factor_overlay.png: 4 curves on one axis (baseline, ridge, projection, noiseless)
- factor_bars_at_N.png: log-scale bar chart of MSE-at-P=N for all four conditions
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

# colors for the three factors + baseline
COLOR_BASE = "#c0392b"      # red
COLOR_RIDGE = "#1f77b4"     # blue
COLOR_PROJ = "#ff7f0e"      # orange
COLOR_NOISE = "#2ca02c"     # green


def _agg(rows, key, val):
    g = {}
    for r in rows:
        g.setdefault(r[key], []).append(r[val])
    xs = sorted(g.keys())
    means = np.array([np.mean(g[x]) for x in xs])
    stds = np.array([np.std(g[x]) for x in xs])
    return np.array(xs), means, stds


def fig_regime_annotated():
    p = RUNS_DIR / "random_feature_ridge.json"
    if not p.exists():
        return
    with open(p) as f:
        data = json.load(f)
    n = data["n_train"]
    rows = data["rows"]
    xs, mtest, stest = _agg(rows, "p", "test_mse")
    _, mtrain, _ = _agg(rows, "p", "train_mse")

    fig, ax = plt.subplots(figsize=(7.6, 4.0))

    # shaded regimes
    ax.axvspan(min(xs), n * 0.85, color="#cfe2f3", alpha=0.5, zorder=0)
    ax.axvspan(n * 0.85, n * 1.5, color="#f4cccc", alpha=0.5, zorder=0)
    ax.axvspan(n * 1.5, max(xs), color="#d9ead3", alpha=0.5, zorder=0)

    ax.plot(xs, mtrain, marker="o", color=COLOR_RIDGE, label="train MSE",
            linewidth=1.7, markersize=5)
    ax.fill_between(xs, mtrain - 0, mtrain + 0, color=COLOR_RIDGE, alpha=0.15)
    ax.plot(xs, mtest, marker="s", color=COLOR_BASE, label="test MSE",
            linewidth=1.7, markersize=5)
    ax.fill_between(xs, mtest - stest, mtest + stest, color=COLOR_BASE, alpha=0.15)

    ax.axvline(n, linestyle="--", color="gray", linewidth=1.1, alpha=0.8)
    peak_p = xs[np.argmax(mtest)]
    peak_v = mtest.max()
    ax.annotate(f"peak {peak_v:.2f}\nat P={peak_p}",
                xy=(peak_p, peak_v),
                xytext=(peak_p * 2.6, peak_v * 0.85),
                fontsize=9, color="black",
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.8))

    ax.text(min(xs) * 1.4, ax.get_ylim()[1] * 0.93, "underparameterized",
            fontsize=9, color="#1a4f7a")
    ax.text(n * 0.92, ax.get_ylim()[1] * 0.93, f"P=N={n}",
            fontsize=9, color="#923c3c", ha="left")
    ax.text(n * 4, ax.get_ylim()[1] * 0.93, "overparameterized",
            fontsize=9, color="#3c6e3c")

    ax.set_xscale("log")
    ax.set_xlabel("number of random features P")
    ax.set_ylabel("MSE")
    ax.set_title("Double descent on California Housing")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    out = FIG_DIR / "regime_annotated.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


def fig_factor_overlay():
    p = RUNS_DIR / "three_factor_ablation.json"
    if not p.exists():
        return
    with open(p) as f:
        data = json.load(f)
    n = data["n_train"]

    fig, ax = plt.subplots(figsize=(8.0, 4.2))

    # baseline (no ridge): factor_a_regularization rows with ablation == baseline_no_ridge
    base_rows = [r for r in data["factor_a_regularization"]
                 if r["ablation"] == "baseline_no_ridge"]
    xs, base_m, _ = _agg(base_rows, "p", "test_mse")

    # cap baseline curve to a reasonable y-axis (otherwise the spike dominates)
    base_clipped = np.clip(base_m, None, 30.0)
    ax.plot(xs, base_clipped, marker="o", color=COLOR_BASE,
            label="baseline (no ridge)", linewidth=1.8, markersize=5)

    # factor (a) at lambda=0.1
    a_rows = [r for r in data["factor_a_regularization"]
              if r["ablation"] == "ridge_lam=0.1"]
    xs_a, a_m, _ = _agg(a_rows, "p", "test_mse")
    ax.plot(xs_a, a_m, marker="s", color=COLOR_RIDGE,
            label="(a) ridge $\\lambda=0.1$", linewidth=1.8, markersize=5)

    # factor (b) at k=N/4
    target_k = n // 4
    b_rows = [r for r in data["factor_b_projection"]
              if r["ablation"] == f"proj_k={target_k}"]
    if b_rows:
        xs_b, b_m, _ = _agg(b_rows, "p", "test_mse")
        ax.plot(xs_b, b_m, marker="^", color=COLOR_PROJ,
                label="(b) projection $k$=" + str(target_k),
                linewidth=1.8, markersize=5)

    # factor (c) noiseless target
    c_rows = data["factor_c_noiseless"]
    xs_c, c_m, _ = _agg(c_rows, "p", "test_mse")
    ax.plot(xs_c, c_m, marker="d", color=COLOR_NOISE,
            label="(c) noiseless target", linewidth=1.8, markersize=5)

    ax.axvline(n, linestyle="--", color="gray", linewidth=1.0, alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of random features P")
    ax.set_ylabel("test MSE (log scale, baseline clipped at 30)")
    ax.set_title("Each factor alone collapses the spike")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.25, which="both")
    plt.tight_layout()
    out = FIG_DIR / "factor_overlay.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


def fig_factor_bars_at_N():
    p = RUNS_DIR / "three_factor_ablation.json"
    if not p.exists():
        return
    with open(p) as f:
        data = json.load(f)
    n = data["n_train"]

    def at_N(rows, ablation_name):
        vs = [r["test_mse"] for r in rows
              if r["ablation"] == ablation_name and r["p"] == n]
        return float(np.mean(vs)) if vs else None

    base_v = at_N(data["factor_a_regularization"], "baseline_no_ridge")
    a_v = at_N(data["factor_a_regularization"], "ridge_lam=0.1")
    b_v = at_N(data["factor_b_projection"], f"proj_k={n // 4}")
    c_v = at_N(data["factor_c_noiseless"], "noiseless_target_no_ridge")

    proj_label = "(b) projection\n$k=N/4=" + str(n // 4) + "$"
    labels = ["baseline\n(no ridge)", "(a) ridge\n$\\lambda=0.1$",
              proj_label, "(c) noiseless\ntarget"]
    values = [base_v, a_v, b_v, c_v]
    colors = [COLOR_BASE, COLOR_RIDGE, COLOR_PROJ, COLOR_NOISE]

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_ylabel("test MSE at P=N (log scale)")
    ax.set_title(f"Test MSE at the interpolation threshold (P=N={n})")
    for b, v in zip(bars, values):
        if v is None:
            continue
        label = f"{v:.2f}" if v < 100 else f"{v:.1e}"
        ax.text(b.get_x() + b.get_width() / 2, v * 1.15, label,
                ha="center", va="bottom", fontsize=10)
    ax.grid(alpha=0.25, axis="y", which="both")
    plt.tight_layout()
    out = FIG_DIR / "factor_bars_at_N.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


def main():
    fig_regime_annotated()
    fig_factor_overlay()
    fig_factor_bars_at_N()


if __name__ == "__main__":
    main()
