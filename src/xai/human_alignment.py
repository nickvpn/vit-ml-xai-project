import numpy as np
import torch
import torch.nn.functional as F
from src.config import GRID_SIZE
from src.utils.metrics import compute_cc, compute_sim


def resize_to_grid(attribution_map, grid_size=GRID_SIZE):
    # resize any attribution map to the grid size for comparison
    if isinstance(attribution_map, np.ndarray):
        attribution_map = torch.tensor(attribution_map, dtype=torch.float32)

    if attribution_map.dim() == 2:
        attribution_map = attribution_map.unsqueeze(0).unsqueeze(0)
    elif attribution_map.dim() == 3:
        attribution_map = attribution_map.unsqueeze(0)

    resized = F.interpolate(attribution_map, size=(grid_size, grid_size),
                           mode="bilinear", align_corners=False)
    return resized.squeeze().numpy()


def compute_human_alignment(attribution_map, saliency_gt, grid_size=GRID_SIZE):
    # compare an explanation map to human saliency ground truth
    # returns CC and SIM scores
    attr_grid = resize_to_grid(attribution_map, grid_size)
    sal_grid = saliency_gt

    if isinstance(sal_grid, torch.Tensor):
        sal_grid = sal_grid.squeeze().numpy()

    # normalize attribution to non-negative for SIM computation
    attr_norm = attr_grid.copy()
    attr_norm = attr_norm - attr_norm.min()
    attr_norm = attr_norm / (attr_norm.sum() + 1e-8)

    cc = compute_cc(attr_grid, sal_grid)
    sim = compute_sim(attr_norm, sal_grid)

    return {"CC": cc, "SIM": sim}


def batch_human_alignment(attribution_maps, saliency_gts):
    # compute alignment for a batch of samples
    cc_scores = []
    sim_scores = []

    for attr, sal in zip(attribution_maps, saliency_gts):
        result = compute_human_alignment(attr, sal)
        cc_scores.append(result["CC"])
        sim_scores.append(result["SIM"])

    return {
        "CC_mean": float(np.mean(cc_scores)),
        "CC_std": float(np.std(cc_scores)),
        "SIM_mean": float(np.mean(sim_scores)),
        "SIM_std": float(np.std(sim_scores)),
    }
