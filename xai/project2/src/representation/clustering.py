import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from src.config import RUNS_DIR, ACTIVATION_SAMPLES, SEED, DEVICE
from src.seed import set_seed
from src.utils.load_models import load_all_variants
from src.representation.activations import build_val_loader, collect_activations


def cluster_purity(labels_true, labels_pred):
    # max-overlap purity: for each cluster, count its plurality true class
    n = len(labels_true)
    contingency = {}
    for c, t in zip(labels_pred, labels_true):
        contingency.setdefault(c, {})
        contingency[c][t] = contingency[c].get(t, 0) + 1
    correct = 0
    for c, counts in contingency.items():
        correct += max(counts.values())
    return float(correct / n)


def patch_token_clustering(feats_patch_all, n_clusters=20, seed=SEED):
    # feats_patch_all: (n, num_patches, d) -> flatten to (n*num_patches, d)
    n, p, d = feats_patch_all.shape
    flat = feats_patch_all.reshape(n * p, d)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=5)
    labels = km.fit_predict(flat)
    return labels.reshape(n, p), km


def saliency_label_per_patch(sal_full, num_patches=196, grid=14):
    # convert full-resolution saliency to per-patch quartile bin (0..3)
    # sal_full: (n, 1, 224, 224)
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(sal_full).float()
    coarse = F.adaptive_avg_pool2d(t, grid).numpy()  # (n, 1, 14, 14)
    coarse = coarse.reshape(coarse.shape[0], -1)  # (n, 196)
    # per-image quartile binning
    binned = np.zeros_like(coarse, dtype=np.int32)
    for i in range(coarse.shape[0]):
        q1, q2, q3 = np.quantile(coarse[i], [0.25, 0.5, 0.75])
        b = np.where(coarse[i] < q1, 0,
            np.where(coarse[i] < q2, 1,
            np.where(coarse[i] < q3, 2, 3)))
        binned[i] = b
    return binned


def category_label_per_patch(y_cls, num_patches=196):
    # very coarse: assign each patch the dominant category present in the image
    # for purity vs class label: convert to "primary class" per image
    # multi-label -> first positive class index (0..k-1) or -1 if none
    n, k = y_cls.shape
    primary = np.argmax(y_cls, axis=1)  # may pick 0 if all zeros, mark with sum check
    has_label = y_cls.sum(axis=1) > 0
    primary = np.where(has_label, primary, -1)
    # broadcast across all patches
    return np.tile(primary[:, None], (1, num_patches))


def main(n_samples=ACTIVATION_SAMPLES, layer=11, n_clusters=20):
    set_seed(SEED)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"clustering patch tokens at layer {layer} for three variants...")
    models = load_all_variants(device=DEVICE)
    loader = build_val_loader(n_samples=n_samples)

    results = {}
    for mode, model in models.items():
        print(f"\n=== variant: {mode} ===")
        feats, meta = collect_activations(model, loader, device=DEVICE,
                                          layers=[layer], token="patch_all")
        f = feats[layer]  # (n, 196, d)
        sal_lab = saliency_label_per_patch(meta["sal_full"], num_patches=f.shape[1])
        cat_lab = category_label_per_patch(meta["y_cls"], num_patches=f.shape[1])

        labels_pred, _ = patch_token_clustering(f, n_clusters=n_clusters, seed=SEED)

        flat_pred = labels_pred.reshape(-1)
        flat_sal = sal_lab.reshape(-1)
        flat_cat = cat_lab.reshape(-1)

        # filter out images with no class label for cat-purity
        mask = flat_cat >= 0
        cat_purity = cluster_purity(flat_cat[mask], flat_pred[mask])
        cat_nmi = float(normalized_mutual_info_score(flat_cat[mask], flat_pred[mask]))
        sal_purity = cluster_purity(flat_sal, flat_pred)
        sal_nmi = float(normalized_mutual_info_score(flat_sal, flat_pred))
        sal_ari = float(adjusted_rand_score(flat_sal, flat_pred))

        # stability: run a second clustering with a different seed and measure ARI between assignments
        labels_pred2, _ = patch_token_clustering(f, n_clusters=n_clusters, seed=SEED + 1)
        stab = float(adjusted_rand_score(labels_pred.reshape(-1), labels_pred2.reshape(-1)))

        results[mode] = {
            "category_purity": cat_purity,
            "category_nmi": cat_nmi,
            "saliency_quartile_purity": sal_purity,
            "saliency_quartile_nmi": sal_nmi,
            "saliency_quartile_ari": sal_ari,
            "seed_stability_ari": stab,
            "layer": layer,
            "n_clusters": n_clusters,
        }
        print(f"category purity={cat_purity:.3f} nmi={cat_nmi:.3f}")
        print(f"saliency-quartile purity={sal_purity:.3f} nmi={sal_nmi:.3f} ari={sal_ari:.3f}")
        print(f"seed-stability ari={stab:.3f}")

    out = RUNS_DIR / "clustering.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nresults saved to {out}")
    return results


if __name__ == "__main__":
    main()
