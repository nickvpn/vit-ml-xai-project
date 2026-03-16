import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image

from src.config import NUM_LABELS


class SaliconCocoDataset(Dataset):
    def __init__(self, img_dir, saliency_dir, coco_ann_path, split_prefix,
                 img_tfm=None, sal_tfm=None, k_labels=NUM_LABELS,
                 cat_to_idx=None, cat_names=None):
        self.img_dir = Path(img_dir)
        self.saliency_dir = Path(saliency_dir)
        self.img_tfm = img_tfm
        self.sal_tfm = sal_tfm
        self.split_prefix = split_prefix  # e.g. "COCO_train2014"

        # load coco annotations
        with open(coco_ann_path, "r") as f:
            coco = json.load(f)

        # build image_id -> set of category ids
        img_to_cats = {}
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            cat_id = ann["category_id"]
            if img_id not in img_to_cats:
                img_to_cats[img_id] = set()
            img_to_cats[img_id].add(cat_id)

        # build category id -> name mapping
        cat_id_to_name = {}
        for cat_info in coco["categories"]:
            cat_id_to_name[cat_info["id"]] = cat_info["name"]

        if cat_to_idx is not None:
            # use provided category mapping (for val/test consistency)
            self.cat_to_idx = cat_to_idx
            self.cat_names = cat_names
        else:
            # top-k most frequent categories from this split
            cat_counts = {}
            for cats in img_to_cats.values():
                for c in cats:
                    cat_counts[c] = cat_counts.get(c, 0) + 1
            top_cats = sorted(cat_counts, key=cat_counts.get, reverse=True)[:k_labels]
            self.cat_to_idx = {c: i for i, c in enumerate(top_cats)}
            self.cat_names = [cat_id_to_name[c] for c in top_cats]

        self.k_labels = len(self.cat_to_idx)
        self.img_to_cats = img_to_cats

        # collect image ids that exist in both the image dir and saliency dir
        self.samples = []
        for img_file in sorted(self.img_dir.glob("*.jpg")):
            # extract numeric id from filename like COCO_train2014_000000000009.jpg
            stem = img_file.stem  # COCO_train2014_000000000009
            img_id = int(stem.split("_")[-1])

            sal_file = self.saliency_dir / f"{stem}.png"
            if sal_file.exists():
                self.samples.append((img_file, sal_file, img_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img_path, sal_path, img_id = self.samples[i]

        img = Image.open(img_path).convert("RGB")
        sal = Image.open(sal_path).convert("L")

        if self.img_tfm is not None:
            img = self.img_tfm(img)

        if self.sal_tfm is not None:
            sal_full, sal_grid = self.sal_tfm(sal)
        else:
            # fallback, raw tensor
            from torchvision.transforms import ToTensor
            sal_full = ToTensor()(sal)
            sal_grid = sal_full

        # multi-label vector
        y = torch.zeros(self.k_labels, dtype=torch.float32)
        cats = self.img_to_cats.get(img_id, set())
        for c in cats:
            if c in self.cat_to_idx:
                y[self.cat_to_idx[c]] = 1.0

        return {
            "image": img,
            "y_cls": y,
            "sal_full": sal_full,
            "sal_grid": sal_grid,
            "img_id": img_id,
        }
