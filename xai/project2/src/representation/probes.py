import json
import numpy as np
import torch
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.metrics import average_precision_score, r2_score
from sklearn.model_selection import train_test_split

from src.config import RUNS_DIR, NUM_BLOCKS, GRID_SIZE, ACTIVATION_SAMPLES, SEED, DEVICE
from src.seed import set_seed
from src.utils.load_models import load_all_variants
from src.representation.activations import build_val_loader, collect_activations


def _split(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=SEED)


def fit_class_probe(X_train, y_train, X_test, y_test):
    # multi-label probe: one logistic regression per label, mean AP across labels
    aps = []
    for i in range(y_train.shape[1]):
        if y_train[:, i].sum() == 0:
            continue
        clf = LogisticRegression(max_iter=200, C=1.0, solver="liblinear")
        clf.fit(X_train, y_train[:, i])
        scores = clf.decision_function(X_test)
        aps.append(average_precision_score(y_test[:, i], scores))
    return float(np.mean(aps)) if aps else 0.0


def fit_saliency_probe(X_train, sal_train, X_test, sal_test):
    # saliency target is per-image full-resolution map flattened to a fixed length;
    # we predict a downsampled grid and report R^2 on the held-out flattened grid
    # ridge-cv with a small alpha grid
    alphas = [0.1, 1.0, 10.0, 100.0]
    reg = RidgeCV(alphas=alphas)
    reg.fit(X_train, sal_train)
    pred = reg.predict(X_test)
    return float(r2_score(sal_test.reshape(sal_test.shape[0], -1).reshape(-1),
                           pred.reshape(-1)))


def downsample_saliency(sal_full, grid=GRID_SIZE):
    # sal_full: (n, 1, 224, 224). reduce to (n, grid*grid) by area-pool average
    import torch.nn.functional as F
    t = torch.from_numpy(sal_full).float()
    g = F.adaptive_avg_pool2d(t, grid).reshape(t.shape[0], -1).numpy()
    return g


def object_location_label(sal_full):
    # convert saliency map to a coarse 4-cell location label (top-left, top-right, bottom-left, bottom-right)
    # uses the centroid of the saliency map
    n = sal_full.shape[0]
    H = sal_full.shape[2]
    W = sal_full.shape[3]
    flat = sal_full.reshape(n, -1)
    flat = flat / (flat.sum(axis=1, keepdims=True) + 1e-8)
    ys = np.arange(H).reshape(1, H, 1)
    xs = np.arange(W).reshape(1, 1, W)
    cy = (sal_full[:, 0] * ys).sum(axis=(1, 2)) / (sal_full[:, 0].sum(axis=(1, 2)) + 1e-8)
    cx = (sal_full[:, 0] * xs).sum(axis=(1, 2)) / (sal_full[:, 0].sum(axis=(1, 2)) + 1e-8)
    top = (cy < H / 2)
    left = (cx < W / 2)
    label = np.where(top & left, 0,
            np.where(top & ~left, 1,
            np.where(~top & left, 2, 3)))
    return label


def fit_location_probe(X_train, l_train, X_test, l_test):
    clf = LogisticRegression(max_iter=300, C=1.0, multi_class="multinomial",
                              solver="lbfgs")
    clf.fit(X_train, l_train)
    return float(clf.score(X_test, l_test))


def random_label_control(X_train, X_test, n_classes=4):
    # check probe's intrinsic capacity by predicting random labels
    rng = np.random.RandomState(SEED)
    y_train = rng.randint(0, n_classes, size=X_train.shape[0])
    y_test = rng.randint(0, n_classes, size=X_test.shape[0])
    clf = LogisticRegression(max_iter=300, C=1.0, multi_class="multinomial",
                              solver="lbfgs")
    clf.fit(X_train, y_train)
    return float(clf.score(X_test, y_test))


def main(n_samples=ACTIVATION_SAMPLES):
    set_seed(SEED)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"loading variants and collecting activations on {n_samples} val samples...")
    models = load_all_variants(device=DEVICE)
    loader = build_val_loader(n_samples=n_samples)

    all_results = {}

    for mode, model in models.items():
        print(f"\n=== variant: {mode} ===")
        feats, meta = collect_activations(model, loader, device=DEVICE,
                                          layers=list(range(NUM_BLOCKS)),
                                          token="cls")
        y = meta["y_cls"]
        sal_full = meta["sal_full"]
        sal_grid = downsample_saliency(sal_full)
        loc_label = object_location_label(sal_full)

        layer_results = {"layers": [], "class_map": [], "saliency_r2": [],
                         "location_acc": [], "random_control_acc": []}

        for li in range(NUM_BLOCKS):
            X = feats[li]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=SEED
            )
            _, _, s_train, s_test = train_test_split(
                X, sal_grid, test_size=0.2, random_state=SEED
            )
            _, _, l_train, l_test = train_test_split(
                X, loc_label, test_size=0.2, random_state=SEED
            )

            cmap = fit_class_probe(X_train, y_train, X_test, y_test)
            sr2 = fit_saliency_probe(X_train, s_train, X_test, s_test)
            lacc = fit_location_probe(X_train, l_train, X_test, l_test)
            rctl = random_label_control(X_train, X_test, n_classes=4)

            layer_results["layers"].append(li)
            layer_results["class_map"].append(cmap)
            layer_results["saliency_r2"].append(sr2)
            layer_results["location_acc"].append(lacc)
            layer_results["random_control_acc"].append(rctl)
            print(f"layer {li:2d} | mAP={cmap:.3f} | sal_R2={sr2:.3f} | "
                  f"loc_acc={lacc:.3f} | rand_ctrl={rctl:.3f}")

        all_results[mode] = layer_results

    out = RUNS_DIR / "probes.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nresults saved to {out}")
    return all_results


if __name__ == "__main__":
    main()
