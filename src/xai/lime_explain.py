import torch
import numpy as np
from lime import lime_image
from src.config import GRID_SIZE


def lime_explain(model, image, target_class, device, num_samples=500):
    # lime explanation using patch-grid segmentation
    # image: (1, 3, 224, 224) tensor
    model.eval()

    img_np = image[0].cpu().permute(1, 2, 0).numpy()

    def predict_fn(images):
        # images: numpy array (n, h, w, 3)
        batch = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
        with torch.no_grad():
            logits, _ = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    # use patch grid as segmentation
    h, w = img_np.shape[:2]
    patch_h = h // GRID_SIZE
    patch_w = w // GRID_SIZE
    segments = np.zeros((h, w), dtype=int)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            segments[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w] = i * GRID_SIZE + j

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_np,
        predict_fn,
        top_labels=None,
        labels=(target_class,),
        num_samples=num_samples,
        segmentation_fn=lambda x: segments,
    )

    # get the importance map
    local_exp = explanation.local_exp.get(target_class, [])

    importance_grid = np.zeros(GRID_SIZE * GRID_SIZE)
    for seg_id, weight in local_exp:
        importance_grid[seg_id] = weight

    importance_grid = importance_grid.reshape(GRID_SIZE, GRID_SIZE)

    return importance_grid
