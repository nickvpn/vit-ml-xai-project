import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    SALICON_DIR, COCO_ANN_DIR, RUNS_DIR,
    BATCH_SIZE, DEVICE, NUM_LABELS, VIT_MODEL_NAME
)
from src.seed import set_seed
from src.data.salicon_coco import SaliconCocoDataset
from src.data.transforms import build_image_transform, SaliencyTransform
from src.utils.metrics import compute_map, compute_cc, compute_sim
import timm


def extract_features(model, loader, device):
    # extract frozen vit features for all samples
    model.eval()
    all_cls_feats = []
    all_patch_feats = []
    all_labels = []
    all_sal_grids = []

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            feats = model.forward_features(x)

            cls_tok = feats[:, 0, :].cpu().numpy()
            patch_toks = feats[:, 1:, :].cpu().numpy()

            all_cls_feats.append(cls_tok)
            # average patch tokens for a global feature
            all_patch_feats.append(patch_toks.mean(axis=1))
            all_labels.append(batch["y_cls"].numpy())
            all_sal_grids.append(batch["sal_grid"].numpy())

    cls_feats = np.concatenate(all_cls_feats, axis=0)
    patch_feats = np.concatenate(all_patch_feats, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    sal_grids = np.concatenate(all_sal_grids, axis=0)

    return cls_feats, patch_feats, labels, sal_grids


def run_classification_baselines(train_feats, train_labels, val_feats, val_labels):
    results = {}

    # logistic regression, one per label (ovr)
    print("  logistic regression...")
    lr_preds = np.zeros_like(val_labels)
    for i in range(train_labels.shape[1]):
        if train_labels[:, i].sum() < 2:
            continue
        clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
        clf.fit(train_feats, train_labels[:, i])
        lr_preds[:, i] = clf.predict_proba(val_feats)[:, 1] if len(clf.classes_) > 1 else 0.0
    results["logistic_regression"] = {"mAP": compute_map(val_labels, lr_preds)}
    print(f"    mAP = {results['logistic_regression']['mAP']:.4f}")

    # small mlp
    print("  mlp classifier...")
    mlp_preds = np.zeros_like(val_labels)
    for i in range(train_labels.shape[1]):
        if train_labels[:, i].sum() < 2:
            continue
        clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=500, early_stopping=True)
        clf.fit(train_feats, train_labels[:, i])
        mlp_preds[:, i] = clf.predict_proba(val_feats)[:, 1] if len(clf.classes_) > 1 else 0.0
    results["mlp_classifier"] = {"mAP": compute_map(val_labels, mlp_preds)}
    print(f"    mAP = {results['mlp_classifier']['mAP']:.4f}")

    return results


def run_saliency_baselines(train_feats, train_sal, val_feats, val_sal):
    results = {}

    # flatten saliency grids
    n_train = train_sal.shape[0]
    n_val = val_sal.shape[0]
    train_sal_flat = train_sal.reshape(n_train, -1)
    val_sal_flat = val_sal.reshape(n_val, -1)

    # mean saliency baseline
    print("  mean saliency baseline...")
    mean_sal = train_sal_flat.mean(axis=0)
    cc_scores = []
    sim_scores = []
    for i in range(n_val):
        cc_scores.append(compute_cc(mean_sal, val_sal_flat[i]))
        sim_scores.append(compute_sim(mean_sal, val_sal_flat[i]))
    results["mean_saliency"] = {
        "CC": float(np.mean(cc_scores)),
        "SIM": float(np.mean(sim_scores)),
    }
    print(f"    CC = {results['mean_saliency']['CC']:.4f}, SIM = {results['mean_saliency']['SIM']:.4f}")

    # ridge regression
    print("  ridge regression...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(train_feats, train_sal_flat)
    ridge_preds = ridge.predict(val_feats)
    cc_scores = []
    sim_scores = []
    for i in range(n_val):
        cc_scores.append(compute_cc(ridge_preds[i], val_sal_flat[i]))
        sim_scores.append(compute_sim(ridge_preds[i], val_sal_flat[i]))
    results["ridge_regression"] = {
        "CC": float(np.mean(cc_scores)),
        "SIM": float(np.mean(sim_scores)),
    }
    print(f"    CC = {results['ridge_regression']['CC']:.4f}, SIM = {results['ridge_regression']['SIM']:.4f}")

    # mlp regressor
    print("  mlp regressor...")
    mlp_reg = MLPRegressor(hidden_layer_sizes=(256,), max_iter=500, early_stopping=True)
    mlp_reg.fit(train_feats, train_sal_flat)
    mlp_preds = mlp_reg.predict(val_feats)
    cc_scores = []
    sim_scores = []
    for i in range(n_val):
        cc_scores.append(compute_cc(mlp_preds[i], val_sal_flat[i]))
        sim_scores.append(compute_sim(mlp_preds[i], val_sal_flat[i]))
    results["mlp_regressor"] = {
        "CC": float(np.mean(cc_scores)),
        "SIM": float(np.mean(sim_scores)),
    }
    print(f"    CC = {results['mlp_regressor']['CC']:.4f}, SIM = {results['mlp_regressor']['SIM']:.4f}")

    return results


def main():
    set_seed(42)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    print("loading frozen vit backbone for feature extraction...")
    backbone = timm.create_model(VIT_MODEL_NAME, pretrained=True, num_classes=0)
    backbone = backbone.to(DEVICE)
    backbone.eval()

    img_tfm = build_image_transform(train=False)
    sal_tfm = SaliencyTransform()

    train_ds = SaliconCocoDataset(
        img_dir=SALICON_DIR / "images" / "train",
        saliency_dir=SALICON_DIR / "train",
        coco_ann_path=COCO_ANN_DIR / "instances_train2014.json",
        split_prefix="COCO_train2014",
        img_tfm=img_tfm, sal_tfm=sal_tfm, k_labels=NUM_LABELS,
    )
    val_ds = SaliconCocoDataset(
        img_dir=SALICON_DIR / "images" / "val",
        saliency_dir=SALICON_DIR / "val",
        coco_ann_path=COCO_ANN_DIR / "instances_val2014.json",
        split_prefix="COCO_val2014",
        img_tfm=img_tfm, sal_tfm=sal_tfm, k_labels=NUM_LABELS,
        cat_to_idx=train_ds.cat_to_idx,
        cat_names=train_ds.cat_names,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    print("extracting train features...")
    train_cls, train_patch, train_labels, train_sal = extract_features(backbone, train_loader, DEVICE)
    print(f"  cls features: {train_cls.shape}, labels: {train_labels.shape}, sal: {train_sal.shape}")

    print("extracting val features...")
    val_cls, val_patch, val_labels, val_sal = extract_features(backbone, val_loader, DEVICE)

    print("\n--- classification baselines ---")
    cls_results = run_classification_baselines(train_cls, train_labels, val_cls, val_labels)

    print("\n--- saliency baselines ---")
    sal_results = run_saliency_baselines(train_cls, train_sal, val_cls, val_sal)

    all_results = {"classification": cls_results, "saliency": sal_results}
    with open(RUNS_DIR / "baseline_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nbaseline results saved to runs/baseline_results.json")


if __name__ == "__main__":
    main()
