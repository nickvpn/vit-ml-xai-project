import json
import numpy as np
from sklearn.metrics import mean_squared_error

from src.config import (
    RUNS_DIR, P_VALUES, N_FEATURE_SEEDS, DEFAULT_RIDGE, SIGMA, SEED
)
from src.seed import set_seed
from src.data.load import load_dataset, make_splits
from src.features.random_features import (
    median_heuristic, sample_random_features_rbf, random_feature_map
)


def random_feature_ridge_fit(X_tr, y_tr, X_test, y_test, p, sigma, seed,
                              alpha=DEFAULT_RIDGE):
    d = X_tr.shape[1]
    omega, b = sample_random_features_rbf(d, p, sigma, seed=seed)
    Phi_tr = random_feature_map(X_tr, omega, b)
    Phi_te = random_feature_map(X_test, omega, b)

    # solve (Phi^T Phi + alpha I) w = Phi^T y; this is the underparam form.
    # but for p > n it is faster to solve the dual form; we still use the same
    # closed form because numpy handles it fine for these sizes.
    n = Phi_tr.shape[0]
    if p < n:
        A = Phi_tr.T @ Phi_tr + alpha * np.eye(p)
        Bt = Phi_tr.T @ y_tr
        w = np.linalg.solve(A, Bt)
    else:
        # dual: w = Phi^T (Phi Phi^T + alpha I)^-1 y
        K = Phi_tr @ Phi_tr.T + alpha * np.eye(n)
        a = np.linalg.solve(K, y_tr)
        w = Phi_tr.T @ a
    pred_tr = Phi_tr @ w
    pred_te = Phi_te @ w
    return {
        "p": int(p),
        "seed": int(seed),
        "train_mse": float(mean_squared_error(y_tr, pred_tr)),
        "test_mse": float(mean_squared_error(y_test, pred_te)),
        "weight_norm": float(np.linalg.norm(w)),
    }


def main():
    set_seed(SEED)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    X, y = load_dataset()
    X_tr, y_tr, X_test, y_test, _, _ = make_splits(X, y)
    sigma = SIGMA if SIGMA is not None else median_heuristic(X_tr)
    print(f"n_train={len(X_tr)} sigma={sigma:.4f}")

    rows = []
    for p in P_VALUES:
        for sd in range(N_FEATURE_SEEDS):
            seed_eff = SEED + sd * 1000
            r = random_feature_ridge_fit(X_tr, y_tr, X_test, y_test, p, sigma,
                                          seed=seed_eff, alpha=DEFAULT_RIDGE)
            rows.append(r)
            print(f"p={p:5d} seed={sd} train={r['train_mse']:.4f} "
                  f"test={r['test_mse']:.4f} ||w||={r['weight_norm']:.3f}")

    out = {"sigma_rbf": sigma, "n_train": len(X_tr), "lambda": DEFAULT_RIDGE,
           "rows": rows}
    p = RUNS_DIR / "random_feature_ridge.json"
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {p}")


if __name__ == "__main__":
    main()
