import sys
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    SALICON_DIR, COCO_ANN_DIR, RUNS_DIR,
    DEVICE, NUM_LABELS, GRID_SIZE
)
from src.seed import set_seed
from src.data.salicon_coco import SaliconCocoDataset
from src.data.transforms import build_image_transform, SaliencyTransform
from src.models.vit_multitask import build_model
from src.utils.io import load_checkpoint
from src.utils.viz import plot_explanation_comparison, denormalize_image

from src.xai.gradient_saliency import compute_gradient_saliency, compute_grad_x_input
from src.xai.ig_explain import integrated_gradients
from src.xai.lime_explain import lime_explain
from src.xai.attention_rollout import attention_rollout
from src.xai.faithfulness import deletion_test, compute_deletion_auc, insertion_test
from src.xai.stability import stability_test
from src.xai.human_alignment import compute_human_alignment, resize_to_grid


def get_top_class(model, image, device):
    # get the most confident predicted class
    model.eval()
    with torch.no_grad():
        logits, _ = model(image.to(device))
        probs = torch.sigmoid(logits[0])
    return int(probs.argmax().item())


def run_explanations_single(model, image, target_class, device):
    # run all explanation methods on a single image
    results = {}

    # gradient saliency
    grad_sal = compute_gradient_saliency(model, image, target_class, device)
    results["gradient_saliency"] = resize_to_grid(grad_sal, GRID_SIZE)

    # gradient x input
    grad_inp = compute_grad_x_input(model, image, target_class, device)
    results["grad_x_input"] = resize_to_grid(grad_inp, GRID_SIZE)

    # integrated gradients
    ig = integrated_gradients(model, image, target_class, device, steps=50)
    results["integrated_gradients"] = resize_to_grid(ig, GRID_SIZE)

    # attention rollout
    rollout = attention_rollout(model, image, device)
    results["attention_rollout"] = rollout

    # lime (slower, fewer samples for speed)
    lime_map = lime_explain(model, image, target_class, device, num_samples=300)
    results["lime"] = lime_map

    return results


def main(n_samples=100):
    set_seed(42)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    figures_dir = RUNS_DIR / "figures"
    figures_dir.mkdir(exist_ok=True)

    print("loading trained model...")
    model, _ = build_model(mode="multitask", num_labels=NUM_LABELS, pretrained=False)
    ckpt_path = RUNS_DIR / "best_multitask.pt"
    load_checkpoint(model, ckpt_path)
    model = model.to(DEVICE)
    model.eval()

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

    # use a subset for xai analysis
    indices = list(range(min(n_samples, len(val_ds))))
    subset = Subset(val_ds, indices)

    print(f"running xai analysis on {len(indices)} samples...")

    method_names = ["gradient_saliency", "grad_x_input", "integrated_gradients",
                    "attention_rollout", "lime"]

    # storage for batch results
    all_faithfulness = {m: [] for m in method_names}
    all_insertion = {m: [] for m in method_names}
    all_stability = {m: [] for m in method_names}
    all_alignment = {m: [] for m in method_names}

    for idx in range(len(indices)):
        sample = subset[idx]
        image = sample["image"].unsqueeze(0)
        sal_grid_gt = sample["sal_grid"]

        target_class = get_top_class(model, image, DEVICE)

        if idx % 10 == 0:
            print(f"  sample {idx+1}/{len(indices)}, target class: {train_ds.cat_names[target_class]}")

        # generate explanations
        explanations = run_explanations_single(model, image, target_class, DEVICE)

        # faithfulness (deletion test)
        for method_name, attr_map in explanations.items():
            fracs, scores = deletion_test(model, image, attr_map, target_class, DEVICE, steps=10)
            del_auc = compute_deletion_auc(fracs, scores)
            all_faithfulness[method_name].append(del_auc)

            ins_fracs, ins_scores = insertion_test(model, image, attr_map, target_class, DEVICE, steps=10)
            ins_auc = compute_deletion_auc(ins_fracs, ins_scores)
            all_insertion[method_name].append(ins_auc)

        # human alignment
        for method_name, attr_map in explanations.items():
            alignment = compute_human_alignment(attr_map, sal_grid_gt, GRID_SIZE)
            all_alignment[method_name].append(alignment)

        # stability (only on gradient-based methods for speed)
        for method_name in ["gradient_saliency", "integrated_gradients"]:
            if method_name == "gradient_saliency":
                def explain_fn(model, img, tc, dev):
                    return resize_to_grid(
                        compute_gradient_saliency(model, img, tc, dev), GRID_SIZE
                    )
            else:
                def explain_fn(model, img, tc, dev):
                    return resize_to_grid(
                        integrated_gradients(model, img, tc, dev, steps=25), GRID_SIZE
                    )

            stab_results, _ = stability_test(explain_fn, model, image, target_class, DEVICE)
            all_stability[method_name].append(stab_results)

        # save a few visual examples
        if idx < 5:
            plot_explanation_comparison(
                sample["image"],
                [explanations[m] for m in method_names],
                method_names,
                save_path=figures_dir / f"xai_comparison_{idx}.png",
            )

    # aggregate results
    summary = {}
    for method_name in method_names:
        summary[method_name] = {
            "deletion_auc_mean": float(np.mean(all_faithfulness[method_name])),
            "deletion_auc_std": float(np.std(all_faithfulness[method_name])),
            "insertion_auc_mean": float(np.mean(all_insertion[method_name])),
            "insertion_auc_std": float(np.std(all_insertion[method_name])),
            "alignment_CC_mean": float(np.mean([a["CC"] for a in all_alignment[method_name]])),
            "alignment_CC_std": float(np.std([a["CC"] for a in all_alignment[method_name]])),
            "alignment_SIM_mean": float(np.mean([a["SIM"] for a in all_alignment[method_name]])),
            "alignment_SIM_std": float(np.std([a["SIM"] for a in all_alignment[method_name]])),
        }

    # stability results (only for gradient-based methods)
    for method_name in ["gradient_saliency", "integrated_gradients"]:
        if all_stability[method_name]:
            perturb_names = list(all_stability[method_name][0].keys())
            for pname in perturb_names:
                vals = [s[pname] for s in all_stability[method_name]]
                summary[method_name][f"stability_{pname}_mean"] = float(np.mean(vals))
                summary[method_name][f"stability_{pname}_std"] = float(np.std(vals))

    with open(RUNS_DIR / "xai_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n--- xai summary ---")
    for method_name, results in summary.items():
        print(f"\n{method_name}:")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")

    print(f"\nresults saved to runs/xai_results.json")
    print(f"figures saved to runs/figures/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=100)
    args = parser.parse_args()
    main(n_samples=args.n_samples)
