import json
import numpy as np
import torch

from src.config import RUNS_DIR, NUM_BLOCKS, ACTIVATION_SAMPLES, SEED, DEVICE
from src.seed import set_seed
from src.utils.load_models import load_all_variants
from src.representation.activations import build_val_loader, collect_activations


# ===== centered kernel alignment =====

def _center(K):
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def linear_kernel(X):
    # X: (n, d). returns (n, n) gram matrix
    return X @ X.T


def rbf_kernel(X, sigma=None):
    sq = np.sum(X * X, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2 * X @ X.T
    d2 = np.maximum(d2, 0.0)
    if sigma is None:
        # median heuristic on positive distances
        triu = d2[np.triu_indices_from(d2, k=1)]
        med = np.median(triu)
        sigma = np.sqrt(med / 2.0) if med > 0 else 1.0
    return np.exp(-d2 / (2.0 * sigma * sigma))


def hsic_biased(K, L):
    Kc = _center(K)
    Lc = _center(L)
    n = K.shape[0]
    return (Kc * Lc).sum() / max(1, (n - 1) ** 2)


def hsic_unbiased(K, L):
    # song et al. 2012 unbiased estimator
    n = K.shape[0]
    if n < 4:
        return hsic_biased(K, L)
    Kt = K - np.diag(np.diag(K))
    Lt = L - np.diag(np.diag(L))
    sum_kt = Kt.sum()
    sum_lt = Lt.sum()
    sum_diag = (Kt * Lt).sum()
    term1 = sum_diag / max(1, n * (n - 3))
    term2 = (sum_kt * sum_lt) / max(1, (n - 1) * (n - 2) * n * (n - 3))
    term3 = -2 * (Kt @ Lt).trace() / max(1, n * (n - 2) * (n - 3))
    return term1 + term2 + term3


def cka(X, Y, kernel="linear", debiased=True):
    if kernel == "linear":
        K = linear_kernel(X)
        L = linear_kernel(Y)
    elif kernel == "rbf":
        K = rbf_kernel(X)
        L = rbf_kernel(Y)
    else:
        raise ValueError(f"unknown kernel: {kernel}")

    hsic = hsic_unbiased if debiased else hsic_biased
    num = hsic(K, L)
    den = np.sqrt(max(0.0, hsic(K, K) * hsic(L, L)))
    if den < 1e-12:
        return 0.0
    return float(num / den)


# ===== procrustes =====

def procrustes_distance(X, Y):
    # orthogonal alignment, returns scaled normalized distance in [0, 1]
    # both X, Y centered and frobenius-normalized
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    Xc = Xc / (np.linalg.norm(Xc) + 1e-12)
    Yc = Yc / (np.linalg.norm(Yc) + 1e-12)

    # singular values of Xc^T Yc give the optimal alignment
    M = Xc.T @ Yc
    s = np.linalg.svd(M, compute_uv=False)
    # similarity in [0,1], higher = more similar
    sim = float(s.sum())
    return sim


# ===== svcca =====

def svcca(X, Y, n_components=20):
    # compute correlations between top-k SVD components of X and Y
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)

    Ux, Sx, _ = np.linalg.svd(Xc, full_matrices=False)
    Uy, Sy, _ = np.linalg.svd(Yc, full_matrices=False)

    k = min(n_components, Ux.shape[1], Uy.shape[1])
    Ux = Ux[:, :k]
    Uy = Uy[:, :k]

    # canonical correlations
    M = Ux.T @ Uy
    rho = np.linalg.svd(M, compute_uv=False)
    # mean of canonical correlations as similarity measure
    return float(np.mean(rho[:k]))


# ===== orchestrator =====

def main(n_samples=ACTIVATION_SAMPLES):
    set_seed(SEED)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"collecting activations across three variants on {n_samples} val samples...")
    models = load_all_variants(device=DEVICE)
    loader = build_val_loader(n_samples=n_samples)

    feats_per_mode = {}
    for mode, model in models.items():
        print(f"variant {mode}...")
        f, _ = collect_activations(model, loader, device=DEVICE,
                                   layers=list(range(NUM_BLOCKS)), token="cls")
        feats_per_mode[mode] = f

    modes = list(feats_per_mode.keys())
    pairs = [("multitask", "cls_only"), ("multitask", "sal_only"),
             ("cls_only", "sal_only")]

    results = {pair: {"layers": [], "cka_linear": [], "cka_rbf": [],
                       "procrustes": [], "svcca": []}
               for pair in [f"{a}_vs_{b}" for a, b in pairs]}

    for li in range(NUM_BLOCKS):
        for a, b in pairs:
            key = f"{a}_vs_{b}"
            Xa = feats_per_mode[a][li]
            Xb = feats_per_mode[b][li]
            cka_l = cka(Xa, Xb, kernel="linear", debiased=True)
            cka_r = cka(Xa, Xb, kernel="rbf", debiased=True)
            proc = procrustes_distance(Xa, Xb)
            svc = svcca(Xa, Xb, n_components=20)
            results[key]["layers"].append(li)
            results[key]["cka_linear"].append(cka_l)
            results[key]["cka_rbf"].append(cka_r)
            results[key]["procrustes"].append(proc)
            results[key]["svcca"].append(svc)
            print(f"layer {li:2d} | {key:30s} | cka_lin={cka_l:.3f} cka_rbf={cka_r:.3f} "
                  f"proc={proc:.3f} svcca={svc:.3f}")

    out = RUNS_DIR / "cka.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nresults saved to {out}")
    return results


if __name__ == "__main__":
    main()
