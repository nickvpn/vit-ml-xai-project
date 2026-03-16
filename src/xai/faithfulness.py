import torch
import numpy as np
import torch.nn.functional as F
from src.config import GRID_SIZE, IMG_SIZE


def deletion_test(model, image, attribution_map, target_class, device, steps=10):
    # progressively mask most important patches and measure prediction drop
    # attribution_map: (14, 14) numpy array
    # returns fractions and corresponding prediction scores
    model.eval()
    image = image.to(device)

    # get baseline prediction
    with torch.no_grad():
        logits, _ = model(image)
        base_score = torch.sigmoid(logits[0, target_class]).item()

    # rank patches by importance (descending)
    flat_attr = attribution_map.flatten()
    ranked_indices = np.argsort(flat_attr)[::-1]  # most important first

    n_patches = len(ranked_indices)
    patch_h = IMG_SIZE // GRID_SIZE
    patch_w = IMG_SIZE // GRID_SIZE

    fractions = []
    scores = []
    fractions.append(0.0)
    scores.append(base_score)

    for step in range(1, steps + 1):
        frac = step / steps
        n_to_mask = int(frac * n_patches)

        masked_image = image.clone()
        for idx in ranked_indices[:n_to_mask]:
            row = idx // GRID_SIZE
            col = idx % GRID_SIZE
            # zero out the patch
            masked_image[0, :,
                        row * patch_h:(row + 1) * patch_h,
                        col * patch_w:(col + 1) * patch_w] = 0.0

        with torch.no_grad():
            logits, _ = model(masked_image)
            score = torch.sigmoid(logits[0, target_class]).item()

        fractions.append(frac)
        scores.append(score)

    return fractions, scores


def compute_deletion_auc(fractions, scores):
    # area under the deletion curve, lower is better (more faithful)
    auc = np.trapz(scores, fractions)
    return float(auc)


def insertion_test(model, image, attribution_map, target_class, device, steps=10):
    # progressively reveal most important patches from a blank image
    model.eval()
    image = image.to(device)

    flat_attr = attribution_map.flatten()
    ranked_indices = np.argsort(flat_attr)[::-1]

    n_patches = len(ranked_indices)
    patch_h = IMG_SIZE // GRID_SIZE
    patch_w = IMG_SIZE // GRID_SIZE

    fractions = []
    scores = []

    # start from blank
    blank = torch.zeros_like(image)
    with torch.no_grad():
        logits, _ = model(blank)
        score = torch.sigmoid(logits[0, target_class]).item()
    fractions.append(0.0)
    scores.append(score)

    for step in range(1, steps + 1):
        frac = step / steps
        n_to_reveal = int(frac * n_patches)

        revealed = blank.clone()
        for idx in ranked_indices[:n_to_reveal]:
            row = idx // GRID_SIZE
            col = idx % GRID_SIZE
            revealed[0, :,
                     row * patch_h:(row + 1) * patch_h,
                     col * patch_w:(col + 1) * patch_w] = image[0, :,
                     row * patch_h:(row + 1) * patch_h,
                     col * patch_w:(col + 1) * patch_w]

        with torch.no_grad():
            logits, _ = model(revealed)
            score = torch.sigmoid(logits[0, target_class]).item()

        fractions.append(frac)
        scores.append(score)

    return fractions, scores
