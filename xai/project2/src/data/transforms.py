import torch
import torchvision.transforms as T
from src.config import IMG_SIZE, GRID_SIZE, IMAGENET_MEAN, IMAGENET_STD


class SaliencyTransform:
    # converts a pil saliency map into full-res and grid tensors
    def __init__(self, img_size=IMG_SIZE, grid_size=GRID_SIZE):
        self.to_tensor = T.ToTensor()
        self.resize_full = T.Resize((img_size, img_size))
        self.resize_grid = T.Resize((grid_size, grid_size))

    def __call__(self, sal_pil):
        sal_full = self.to_tensor(self.resize_full(sal_pil))  # 1x224x224
        sal_grid = self.to_tensor(self.resize_grid(sal_pil))  # 1x14x14

        # normalize to distribution
        eps = 1e-6
        sal_full = sal_full / (sal_full.sum() + eps)
        sal_grid = sal_grid / (sal_grid.sum() + eps)

        return sal_full, sal_grid


def build_image_transform(img_size=IMG_SIZE, train=False):
    # no spatial augmentations, just resize and normalize
    steps = [
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return T.Compose(steps)
