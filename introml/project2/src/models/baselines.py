import json
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.gaussian_process.kernels import RBF

from src.config import RUNS_DIR, DEFAULT_RIDGE, SIGMA, SEED
from src.seed import set_seed
from src.data.load import load_dataset, make_splits
from src.features.random_features import median_heuristic


def fit_linear(X_tr, y_tr, X_test, y_test):
    m = LinearRegression()
    m.fit(X_tr, y_tr)
    p = m.predict(X_test)
    return float(mean_squared_error(y_test, p)), float(r2_score(y_test, p))


def fit_ridge(X_tr, y_tr, X_test, y_test, alpha=DEFAULT_RIDGE):
    m = Ridge(alpha=alpha)
    m.fit(X_tr, y_tr)
    p = m.predict(X_test)
    return float(mean_squared_error(y_test, p)), float(r2_score(y_test, p))


def fit_mean_baseline(y_tr, y_test):
    pred = np.full(y_test.shape, y_tr.mean(), dtype=np.float64)
    return float(mean_squared_error(y_test, pred)), float(r2_score(y_test, pred))


def kernel_ridge_rbf(X_tr, y_tr, X_test, y_test, sigma, alpha=DEFAULT_RIDGE):
    # exact rbf kernel ridge regression as the ground truth p->infty target
    sq_tr = np.sum(X_tr * X_tr, axis=1)
    Ktr = np.exp(-(sq_tr[:, None] + sq_tr[None, :] - 2 * X_tr @ X_tr.T)
                  / (2.0 * sigma * sigma))
    n = Ktr.shape[0]
    a = np.linalg.solve(Ktr + alpha * np.eye(n), y_tr)

    sq_te = np.sum(X_test * X_test, axis=1)
    Kte = np.exp(-(sq_te[:, None] + sq_tr[None, :] - 2 * X_test @ X_tr.T)
                   / (2.0 * sigma * sigma))
    pred = Kte @ a
    return float(mean_squared_error(y_test, pred)), float(r2_score(y_test, pred))


def main():
    set_seed(SEED)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    X, y = load_dataset()
    X_tr, y_tr, X_test, y_test, _, _ = make_splits(X, y)
    sigma = SIGMA if SIGMA is not None else median_heuristic(X_tr)
    print(f"n_train={len(X_tr)} n_test={len(X_test)} d={X.shape[1]} sigma={sigma:.4f}")

    out = {"sigma_rbf": sigma}

    mse, r2 = fit_mean_baseline(y_tr, y_test)
    out["mean"] = {"mse": mse, "r2": r2}
    print(f"mean baseline       mse={mse:.4f} r2={r2:.4f}")

    mse, r2 = fit_linear(X_tr, y_tr, X_test, y_test)
    out["linear"] = {"mse": mse, "r2": r2}
    print(f"linear regression   mse={mse:.4f} r2={r2:.4f}")

    mse, r2 = fit_ridge(X_tr, y_tr, X_test, y_test)
    out["ridge"] = {"mse": mse, "r2": r2}
    print(f"ridge regression    mse={mse:.4f} r2={r2:.4f}")

    mse, r2 = kernel_ridge_rbf(X_tr, y_tr, X_test, y_test, sigma)
    out["kernel_ridge_rbf"] = {"mse": mse, "r2": r2}
    print(f"exact rbf-krr       mse={mse:.4f} r2={r2:.4f}")

    p = RUNS_DIR / "baselines.json"
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {p}")


if __name__ == "__main__":
    main()
