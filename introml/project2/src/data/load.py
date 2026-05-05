import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import DATASET, TEST_SIZE, N_TRAIN, SEED


def load_dataset():
    if DATASET == "california_housing":
        X, y = fetch_california_housing(return_X_y=True)
    else:
        raise ValueError(f"unknown dataset: {DATASET}")
    return X.astype(np.float64), y.astype(np.float64)


def make_splits(X, y, n_train=N_TRAIN, test_size=TEST_SIZE, seed=SEED):
    # global split for the test set; then take a small training subset of size n_train
    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X_pool))[:n_train]
    X_tr = X_pool[idx]
    y_tr = y_pool[idx]

    # standardize features and target on training only
    sx = StandardScaler().fit(X_tr)
    sy = StandardScaler().fit(y_tr.reshape(-1, 1))

    X_tr = sx.transform(X_tr)
    X_test = sx.transform(X_test)
    y_tr = sy.transform(y_tr.reshape(-1, 1)).ravel()
    y_test = sy.transform(y_test.reshape(-1, 1)).ravel()

    return X_tr, y_tr, X_test, y_test, sx, sy
