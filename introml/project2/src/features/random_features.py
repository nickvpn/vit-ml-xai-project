import numpy as np


def median_heuristic(X):
    # median of pairwise distances; standard bandwidth choice for rbf
    n = X.shape[0]
    if n > 1000:
        rng = np.random.RandomState(0)
        idx = rng.choice(n, 1000, replace=False)
        Xs = X[idx]
    else:
        Xs = X
    sq = np.sum(Xs * Xs, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2 * Xs @ Xs.T
    d2 = np.maximum(d2, 0.0)
    triu = d2[np.triu_indices_from(d2, k=1)]
    med = np.median(triu)
    return float(np.sqrt(med / 2.0)) if med > 0 else 1.0


def sample_random_features_rbf(d_in, p, sigma, seed=0):
    # rahimi-recht 2007 cosine features approximating the rbf kernel
    rng = np.random.RandomState(seed)
    omega = rng.normal(scale=1.0 / sigma, size=(d_in, p))
    b = rng.uniform(0.0, 2.0 * np.pi, size=p)
    return omega, b


def random_feature_map(X, omega, b):
    p = omega.shape[1]
    z = X @ omega + b
    return np.sqrt(2.0 / p) * np.cos(z)
