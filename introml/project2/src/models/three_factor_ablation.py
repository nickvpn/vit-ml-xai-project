import json
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge

from src.config import (
    RUNS_DIR, P_VALUES, N_FEATURE_SEEDS, DEFAULT_RIDGE, SIGMA, SEED
)
from src.seed import set_seed
from src.data.load import load_dataset, make_splits
from src.features.random_features import (
    median_heuristic, sample_random_features_rbf, random_feature_map
)
from src.models.random_feature_ridge import random_feature_ridge_fit


# the three factors per schaeffer et al. (2023):
# (a) ridge regularization (lam>0) suppresses small singular values that drive
#     the test-error spike at the interpolation threshold
# (b) projecting test features onto the leading singular components removes
#     trailing-mode variation in the train-feature distribution
# (c) replacing y with the noiseless best-linear-fit removes label noise that
#     amplifies the spike


def fit_baseline(X_tr, y_tr, X_test, y_test, p, sigma, seed):
    return random_feature_ridge_fit(X_tr, y_tr, X_test, y_test, p, sigma,
                                     seed=seed, alpha=0.0)


def fit_with_ridge(X_tr, y_tr, X_test, y_test, p, sigma, seed, alpha):
    return random_feature_ridge_fit(X_tr, y_tr, X_test, y_test, p, sigma,
                                     seed=seed, alpha=alpha)


def fit_with_projection(X_tr, y_tr, X_test, y_test, p, sigma, seed, k):
    # project test features onto the top-k singular directions of the train feature matrix
    d = X_tr.shape[1]
    omega, b = sample_random_features_rbf(d, p, sigma, seed=seed)
    Phi_tr = random_feature_map(X_tr, omega, b)
    Phi_te = random_feature_map(X_test, omega, b)

    Utr, Str, Vtr = np.linalg.svd(Phi_tr, full_matrices=False)
    k_eff = min(k, len(Str))
    # build projection matrix onto leading k right-singular vectors
    Vk = Vtr[:k_eff].T  # (p, k_eff)

    Phi_tr_proj = Phi_tr @ Vk @ Vk.T
    Phi_te_proj = Phi_te @ Vk @ Vk.T

    # least squares with no extra ridge
    n = Phi_tr_proj.shape[0]
    if p < n:
        w = np.linalg.lstsq(Phi_tr_proj, y_tr, rcond=None)[0]
    else:
        K = Phi_tr_proj @ Phi_tr_proj.T + 1e-8 * np.eye(n)
        a = np.linalg.solve(K, y_tr)
        w = Phi_tr_proj.T @ a
    pred_te = Phi_te_proj @ w
    return {
        "p": int(p), "seed": int(seed),
        "test_mse": float(mean_squared_error(y_test, pred_te)),
        "k": int(k_eff),
    }


def fit_with_noiseless_target(X_tr, y_tr, X_test, y_test, p, sigma, seed,
                                alpha=0.0):
    # replace y with best linear projection on raw features (no random features),
    # which acts as a noiseless target (no residuals)
    d = X_tr.shape[1]
    lin = Ridge(alpha=1e-3, fit_intercept=False)
    lin.fit(X_tr, y_tr)
    y_clean = lin.predict(X_tr)
    y_clean_te = lin.predict(X_test)
    return random_feature_ridge_fit(X_tr, y_clean, X_test, y_clean_te,
                                     p, sigma, seed=seed, alpha=alpha)


def smallest_singular_value(X_tr, p, sigma, seed):
    d = X_tr.shape[1]
    omega, b = sample_random_features_rbf(d, p, sigma, seed=seed)
    Phi = random_feature_map(X_tr, omega, b)
    s = np.linalg.svd(Phi, compute_uv=False)
    return float(s[-1])


def main():
    set_seed(SEED)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    X, y = load_dataset()
    X_tr, y_tr, X_test, y_test, _, _ = make_splits(X, y)
    sigma = SIGMA if SIGMA is not None else median_heuristic(X_tr)
    n = len(X_tr)
    print(f"n_train={n} sigma={sigma:.4f}")

    factor_a = []  # different ridge values
    factor_b = []  # leading-mode projection
    factor_c = []  # noiseless targets
    sigmas = []

    for p in P_VALUES:
        for sd in range(N_FEATURE_SEEDS):
            seed_eff = SEED + sd * 1000

            # baseline (no ridge)
            base = random_feature_ridge_fit(X_tr, y_tr, X_test, y_test,
                                              p, sigma, seed=seed_eff, alpha=0.0)
            factor_a.append({**base, "ablation": "baseline_no_ridge"})

            # factor a: ridge regularization at increasing levels
            for alpha in [1e-4, 1e-3, 1e-2, 1e-1]:
                r = random_feature_ridge_fit(X_tr, y_tr, X_test, y_test,
                                              p, sigma, seed=seed_eff,
                                              alpha=alpha)
                factor_a.append({**r, "ablation": f"ridge_lam={alpha}"})

            # factor b: project test features onto leading-k singular directions
            for k in [max(p // 4, 1), max(p // 2, 1)]:
                r = fit_with_projection(X_tr, y_tr, X_test, y_test,
                                          p, sigma, seed=seed_eff, k=k)
                factor_b.append({**r, "ablation": f"proj_k={k}"})

            # factor c: noiseless target
            r = fit_with_noiseless_target(X_tr, y_tr, X_test, y_test,
                                            p, sigma, seed=seed_eff, alpha=0.0)
            factor_c.append({**r, "ablation": "noiseless_target_no_ridge"})

            # smallest singular value of feature matrix (for spectral analysis)
            ss = smallest_singular_value(X_tr, p, sigma, seed=seed_eff)
            sigmas.append({"p": int(p), "seed": int(sd), "smallest_sv": ss})

        print(f"p={p:5d} done")

    out = {
        "factor_a_regularization": factor_a,
        "factor_b_projection": factor_b,
        "factor_c_noiseless": factor_c,
        "smallest_singular_values": sigmas,
        "n_train": n,
        "sigma_rbf": sigma,
    }
    out_path = RUNS_DIR / "three_factor_ablation.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
