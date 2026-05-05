import json
import numpy as np

from src.config import (
    RUNS_DIR, P_VALUES, KPCA_COMPONENTS, SIGMA, SEED
)
from src.seed import set_seed
from src.data.load import load_dataset, make_splits
from src.features.random_features import (
    median_heuristic, sample_random_features_rbf, random_feature_map
)


def kernel_pca_random_features(X_tr, p, sigma, seed, n_components=KPCA_COMPONENTS,
                                  center=False):
    d = X_tr.shape[1]
    omega, b = sample_random_features_rbf(d, p, sigma, seed=seed)
    Phi = random_feature_map(X_tr, omega, b)
    if center:
        Phi = Phi - Phi.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Phi, full_matrices=False)
    k = min(n_components, len(S))
    components = U[:, :k] * S[:k]
    return components, S, U, Vt


def main():
    set_seed(SEED)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    X, y = load_dataset()
    X_tr, y_tr, X_test, y_test, _, _ = make_splits(X, y)
    sigma = SIGMA if SIGMA is not None else median_heuristic(X_tr)
    n = len(X_tr)

    spectra = {}
    for p in [n // 4, n // 2, n, 2 * n, 4 * n, 8 * n, 16 * n]:
        if p < 4 or p > 16384:
            continue
        comps, S, _, _ = kernel_pca_random_features(X_tr, p, sigma, seed=SEED)
        spectra[str(p)] = {
            "p": int(p),
            "singular_values": S.tolist()[:200],
            "smallest": float(S[-1]),
            "ratio_smallest_to_largest": float(S[-1] / max(S[0], 1e-12)),
        }
        print(f"p={p:5d} | smallest sv={S[-1]:.4f} | largest sv={S[0]:.4f} | "
              f"ratio={S[-1] / max(S[0], 1e-12):.4e}")

    out_path = RUNS_DIR / "kernel_pca_spectral.json"
    with open(out_path, "w") as f:
        json.dump({"sigma_rbf": sigma, "n_train": n, "spectra": spectra}, f, indent=2)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
