import torch
import numpy as np


def compute_gradient_saliency(model, image, target_class, device):
    # vanilla gradient saliency map
    # image: (1, 3, 224, 224) tensor
    model.eval()
    image = image.to(device).requires_grad_(True)

    logits, _ = model(image)
    score = logits[0, target_class]
    score.backward()

    grad = image.grad.data[0]  # (3, 224, 224)

    # take abs and max across channels
    saliency = grad.abs().max(dim=0)[0]  # (224, 224)

    return saliency.detach().cpu()


def compute_grad_x_input(model, image, target_class, device):
    # gradient times input attribution
    model.eval()
    image = image.to(device).requires_grad_(True)

    logits, _ = model(image)
    score = logits[0, target_class]
    score.backward()

    grad = image.grad.data[0]  # (3, 224, 224)
    inp = image.data[0]

    attribution = (grad * inp).abs().max(dim=0)[0]  # (224, 224)

    return attribution.detach().cpu()
