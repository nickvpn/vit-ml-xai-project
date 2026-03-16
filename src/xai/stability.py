import torch
import numpy as np
import torchvision.transforms as T
from src.config import IMG_SIZE


def add_gaussian_noise(image, sigma=0.05):
    # add mild gaussian noise to image tensor
    noise = torch.randn_like(image) * sigma
    return image + noise


def horizontal_flip(image):
    return torch.flip(image, dims=[-1])


def slight_brightness(image, factor=0.1):
    return image + factor


def compute_cosine_similarity(map1, map2):
    # cosine similarity between two attribution maps
    v1 = map1.flatten().astype(np.float64)
    v2 = map2.flatten().astype(np.float64)
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1) + 1e-8
    norm2 = np.linalg.norm(v2) + 1e-8
    return float(dot / (norm1 * norm2))


def stability_test(explain_fn, model, image, target_class, device,
                   perturbations=None):
    # test how stable explanations are under small perturbations
    # explain_fn: function(model, image, target_class, device) -> numpy attribution map
    # returns dict of perturbation_name -> cosine_similarity
    if perturbations is None:
        perturbations = {
            "gaussian_noise_0.02": lambda img: add_gaussian_noise(img, 0.02),
            "gaussian_noise_0.05": lambda img: add_gaussian_noise(img, 0.05),
            "horizontal_flip": horizontal_flip,
            "brightness_+0.05": lambda img: slight_brightness(img, 0.05),
        }

    # get baseline explanation
    base_map = explain_fn(model, image, target_class, device)
    if isinstance(base_map, torch.Tensor):
        base_map = base_map.numpy()

    results = {}
    for name, perturb_fn in perturbations.items():
        perturbed = perturb_fn(image.clone())
        perturbed_map = explain_fn(model, perturbed, target_class, device)
        if isinstance(perturbed_map, torch.Tensor):
            perturbed_map = perturbed_map.numpy()

        # for horizontal flip, flip the map back before comparing
        if "flip" in name:
            perturbed_map = np.flip(perturbed_map, axis=-1).copy()

        sim = compute_cosine_similarity(base_map, perturbed_map)
        results[name] = sim

    return results, base_map
