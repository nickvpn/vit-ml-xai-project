import numpy as np
import torch
from sklearn.metrics import average_precision_score


def compute_map(y_true, y_scores):
    # mean average precision for multi-label classification
    # y_true, y_scores: numpy arrays of shape (n_samples, n_labels)
    aps = []
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() > 0:
            ap = average_precision_score(y_true[:, i], y_scores[:, i])
            aps.append(ap)
    if len(aps) == 0:
        return 0.0
    return float(np.mean(aps))


def compute_cc(pred, target):
    # pearson correlation coefficient between two maps
    # pred, target: numpy arrays, flattened
    pred = pred.flatten().astype(np.float64)
    target = target.flatten().astype(np.float64)

    pred = pred - pred.mean()
    target = target - target.mean()

    num = np.sum(pred * target)
    den = np.sqrt(np.sum(pred ** 2) * np.sum(target ** 2)) + 1e-8
    return float(num / den)


def compute_sim(pred, target):
    # similarity metric for saliency maps
    # both should be non-negative and sum to 1
    pred = pred.flatten().astype(np.float64)
    target = target.flatten().astype(np.float64)

    # normalize to distributions
    pred = pred / (pred.sum() + 1e-8)
    target = target / (target.sum() + 1e-8)

    return float(np.sum(np.minimum(pred, target)))


def compute_kl_div(pred, target):
    # kl divergence from target to pred (how much info lost)
    pred = pred.flatten().astype(np.float64)
    target = target.flatten().astype(np.float64)

    # normalize
    pred = pred / (pred.sum() + 1e-8)
    target = target / (target.sum() + 1e-8)

    eps = 1e-8
    kl = np.sum(target * np.log((target + eps) / (pred + eps)))
    return float(kl)
