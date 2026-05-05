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


def _aggregate(rows, key, value_key, group_key="seed"):
    # group rows by `key` (e.g. p), aggregate value_key across `group_key` (e.g. seed)
    groups = {}
    for r in rows:
        groups.setdefault(r[key], []).append(r[value_key])
    xs = sorted(groups.keys())
    ms = [np.mean(groups[x]) for x in xs]
    sds = [np.std(groups[x]) for x in xs]
    return np.array(xs), np.array(ms), np.array(sds)


def fig_double_descent():
    p = RUNS_DIR / "random_feature_ridge.json"
    if not p.exists():
        return
    with open(p) as f:
        data = json.load(f)

    n = data["n_train"]
    rows = data["rows"]
    xs, mtest, stest = _aggregate(rows, "p", "test_mse")
    _, mtrain, strain = _aggregate(rows, "p", "train_mse")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(xs, mtrain, marker="o", color="#1f77b4", label="train MSE")
    ax.fill_between(xs, mtrain - strain, mtrain + strain, color="#1f77b4", alpha=0.15)
    ax.plot(xs, mtest, marker="s", color="#d62728", label="test MSE")
    ax.fill_between(xs, mtest - stest, mtest + stest, color="#d62728", alpha=0.15)
    ax.axvline(n, linestyle="--", color="gray", label=f"P=N={n}")
    ax.set_xscale("log")
    ax.set_xlabel("number of random features P")
    ax.set_ylabel("MSE")
    ax.set_title("Double descent on California Housing")
    ax.grid(alpha=0.3)
    ax.legend()
    out = FIG_DIR / "double_descent.png"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


def fig_three_factor():
    p = RUNS_DIR / "three_factor_ablation.json"
    if not p.exists():
        return
    with open(p) as f:
        data = json.load(f)
    n = data["n_train"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)

    # factor a: ridge sweep
    a_rows = data["factor_a_regularization"]
    by_abl = {}
    for r in a_rows:
        by_abl.setdefault(r["ablation"], []).append(r)
    palette = plt.cm.viridis(np.linspace(0, 1, len(by_abl)))
    for color, abl in zip(palette, sorted(by_abl, key=lambda s: ("baseline" not in s, s))):
        xs, m, s = _aggregate(by_abl[abl], "p", "test_mse")
        axes[0].plot(xs, m, marker="o", color=color, label=abl, markersize=3)
    axes[0].axvline(n, linestyle="--", color="gray")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("P (number of random features)")
    axes[0].set_ylabel("test MSE")
    axes[0].set_title("(a) Ridge regularization")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7, loc="best")

    # factor b: leading-mode projection
    b_rows = data["factor_b_projection"]
    by_abl = {}
    for r in b_rows:
        by_abl.setdefault(r["ablation"], []).append(r)
    palette = plt.cm.plasma(np.linspace(0, 1, len(by_abl)))
    for color, abl in zip(palette, sorted(by_abl)):
        xs, m, s = _aggregate(by_abl[abl], "p", "test_mse")
        axes[1].plot(xs, m, marker="o", color=color, label=abl, markersize=3)
    axes[1].axvline(n, linestyle="--", color="gray")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("P")
    axes[1].set_title("(b) Leading-mode projection")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7, loc="best")

    # factor c: noiseless target
    c_rows = data["factor_c_noiseless"]
    by_abl = {}
    for r in c_rows:
        by_abl.setdefault(r["ablation"], []).append(r)
    for abl, rows in by_abl.items():
        xs, m, s = _aggregate(rows, "p", "test_mse")
        axes[2].plot(xs, m, marker="o", color="C2", label=abl, markersize=3)
    # baseline overlaid
    base_rows = [r for r in a_rows if r["ablation"] == "baseline_no_ridge"]
    xs, m, _ = _aggregate(base_rows, "p", "test_mse")
    axes[2].plot(xs, m, marker="o", color="C3", label="baseline (with noise)", markersize=3,
                  linestyle="--", alpha=0.7)
    axes[2].axvline(n, linestyle="--", color="gray")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("P")
    axes[2].set_title("(c) Noiseless target")
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=7, loc="best")

    plt.tight_layout()
    out = FIG_DIR / "three_factor_ablation.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")

    # smallest singular value plot
    sv_rows = data["smallest_singular_values"]
    by_p = {}
    for r in sv_rows:
        by_p.setdefault(r["p"], []).append(r["smallest_sv"])
    ps = sorted(by_p)
    means = [np.mean(by_p[p]) for p in ps]
    stds = [np.std(by_p[p]) for p in ps]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(ps, means, marker="o", color="#9467bd")
    ax.fill_between(ps, np.array(means) - np.array(stds),
                     np.array(means) + np.array(stds), color="#9467bd", alpha=0.15)
    ax.axvline(n, linestyle="--", color="gray", label=f"P=N={n}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("P")
    ax.set_ylabel("smallest singular value of $\\Phi$")
    ax.set_title("Spectral collapse near the interpolation threshold")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    out = FIG_DIR / "smallest_singular_value.png"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


def fig_kernel_approximation():
    rfp = RUNS_DIR / "random_feature_ridge.json"
    bp = RUNS_DIR / "baselines.json"
    if not (rfp.exists() and bp.exists()):
        return
    with open(rfp) as f:
        rf = json.load(f)
    with open(bp) as f:
        bs = json.load(f)

    rows = rf["rows"]
    xs, m, s = _aggregate(rows, "p", "test_mse")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(xs, m, marker="o", color="C0", label="random-feature ridge")
    ax.fill_between(xs, m - s, m + s, color="C0", alpha=0.15)
    ax.axhline(bs["kernel_ridge_rbf"]["mse"], linestyle="--", color="C2",
                label=f"exact RBF KRR ({bs['kernel_ridge_rbf']['mse']:.3f})")
    ax.axhline(bs["ridge"]["mse"], linestyle=":", color="C3",
                label=f"linear ridge ({bs['ridge']['mse']:.3f})")
    ax.axhline(bs["mean"]["mse"], linestyle=":", color="gray",
                label=f"mean baseline ({bs['mean']['mse']:.3f})")
    ax.set_xscale("log")
    ax.set_xlabel("P")
    ax.set_ylabel("test MSE")
    ax.set_title("Random features approximate exact kernel ridge as P grows")
    ax.legend()
    ax.grid(alpha=0.3)
    out = FIG_DIR / "kernel_approximation.png"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


def fig_kpca_spectrum():
    p = RUNS_DIR / "kernel_pca_spectral.json"
    if not p.exists():
        return
    with open(p) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    palette = plt.cm.cividis(np.linspace(0, 1, len(data["spectra"])))
    for color, (key, rec) in zip(palette, sorted(data["spectra"].items(),
                                                   key=lambda kv: int(kv[0]))):
        sv = rec["singular_values"]
        ax.plot(sv, color=color, label=f"P={rec['p']}")
    ax.set_xlabel("singular value index")
    ax.set_ylabel("singular value")
    ax.set_yscale("log")
    ax.set_title("Random feature singular spectrum at varying P")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3, which="both")
    out = FIG_DIR / "kpca_spectrum.png"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


def main():
    fig_double_descent()
    fig_kernel_approximation()
    fig_three_factor()
    fig_kpca_spectrum()


if __name__ == "__main__":
    main()
