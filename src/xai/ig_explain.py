import torch
import numpy as np


def integrated_gradients(model, image, target_class, device, steps=50, baseline=None):
    # integrated gradients attribution
    # image: (1, 3, 224, 224) tensor
    model.eval()

    if baseline is None:
        baseline = torch.zeros_like(image)

    image = image.to(device)
    baseline = baseline.to(device)

    # scaled inputs along the path from baseline to input
    scaled_inputs = []
    for alpha in torch.linspace(0, 1, steps):
        scaled_input = baseline + alpha * (image - baseline)
        scaled_inputs.append(scaled_input)

    # compute gradients at each step
    grads = []
    for scaled_input in scaled_inputs:
        scaled_input = scaled_input.detach().requires_grad_(True)
        logits, _ = model(scaled_input)
        score = logits[0, target_class]
        model.zero_grad()
        score.backward()
        grads.append(scaled_input.grad.data.clone())

    # average gradients
    avg_grads = torch.stack(grads).mean(dim=0)

    # integrated gradients = (input - baseline) * avg_grads
    ig = (image - baseline) * avg_grads
    ig = ig[0]  # (3, 224, 224)

    # take abs and max across channels for visualization
    attribution = ig.abs().max(dim=0)[0]  # (224, 224)

    return attribution.detach().cpu()
