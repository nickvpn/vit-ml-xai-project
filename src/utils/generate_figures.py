import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import RUNS_DIR


def plot_training_curves_from_history(history_path, save_dir):
    with open(history_path) as f:
        history = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(history["train_loss"], label="train")
    ax1.plot(history["val_loss"], label="val")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_title("training and validation loss")
    ax1.legend()

    ax2.plot(history["val_map"], label="val mAP", color="green")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("mAP")
    ax2.set_title("validation mAP over training")
    ax2.legend()

    # extract mode name from the history path
    mode_name = history_path.stem.replace("history_", "")
    plt.tight_layout()
    plt.savefig(save_dir / f"training_curves_{mode_name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved training_curves_{mode_name}.png")


def plot_baseline_comparison(save_dir):
    # load baseline and model eval results
    baseline_path = RUNS_DIR / "baseline_results.json"
    eval_path = RUNS_DIR / "eval_multitask.json"

    if not baseline_path.exists() or not eval_path.exists():
        print("missing baseline or eval results, skipping comparison plot")
        return

    with open(baseline_path) as f:
        baselines = json.load(f)
    with open(eval_path) as f:
        model_eval = json.load(f)

    # classification comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    cls_methods = list(baselines["classification"].keys())
    cls_maps = [baselines["classification"][m]["mAP"] for m in cls_methods]

    # add all model variants
    for mode_name, color_label in [("cls_only", "ViT cls-only"), ("multitask", "ViT multitask")]:
        mode_path = RUNS_DIR / f"eval_{mode_name}.json"
        if mode_path.exists():
            with open(mode_path) as f2:
                mode_res = json.load(f2)
            cls_methods.append(color_label)
            cls_maps.append(mode_res["mAP"])

    colors = ["#4e79a7"] * (len(cls_methods) - 2) + ["#59a14f", "#e15759"]
    ax1.barh(cls_methods, cls_maps, color=colors)
    ax1.set_xlabel("mAP")
    ax1.set_title("classification: baselines vs fine-tuned")
    ax1.set_xlim(0, 1)

    # saliency comparison
    sal_methods = list(baselines["saliency"].keys())
    sal_ccs = [baselines["saliency"][m]["CC"] for m in sal_methods]

    for mode_name, color_label in [("multitask", "ViT multitask"), ("sal_only", "ViT sal-only")]:
        mode_path = RUNS_DIR / f"eval_{mode_name}.json"
        if mode_path.exists():
            with open(mode_path) as f2:
                mode_res = json.load(f2)
            sal_methods.append(color_label)
            sal_ccs.append(mode_res["CC"])

    colors = ["#4e79a7"] * (len(sal_methods) - 2) + ["#e15759", "#59a14f"]
    ax2.barh(sal_methods, sal_ccs, color=colors)
    ax2.set_xlabel("CC (correlation)")
    ax2.set_title("saliency: baselines vs fine-tuned")
    ax2.set_xlim(-0.1, 1)

    plt.tight_layout()
    plt.savefig(save_dir / "baseline_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved baseline_comparison.png")


def plot_xai_summary(save_dir):
    xai_path = RUNS_DIR / "xai_results.json"
    if not xai_path.exists():
        print("missing xai results, skipping xai plots")
        return

    with open(xai_path) as f:
        xai = json.load(f)

    methods = list(xai.keys())

    # faithfulness plot (deletion + insertion auc)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    del_aucs = [xai[m]["deletion_auc_mean"] for m in methods]
    del_stds = [xai[m]["deletion_auc_std"] for m in methods]
    ax1.barh(methods, del_aucs, xerr=del_stds, color="#4e79a7", capsize=3)
    ax1.set_xlabel("deletion AUC (lower = more faithful)")
    ax1.set_title("faithfulness: deletion test")

    ins_aucs = [xai[m]["insertion_auc_mean"] for m in methods]
    ins_stds = [xai[m]["insertion_auc_std"] for m in methods]
    ax2.barh(methods, ins_aucs, xerr=ins_stds, color="#59a14f", capsize=3)
    ax2.set_xlabel("insertion AUC (higher = more faithful)")
    ax2.set_title("faithfulness: insertion test")

    # human alignment
    align_ccs = [xai[m]["alignment_CC_mean"] for m in methods]
    align_stds = [xai[m]["alignment_CC_std"] for m in methods]
    ax3.barh(methods, align_ccs, xerr=align_stds, color="#f28e2b", capsize=3)
    ax3.set_xlabel("CC with human saliency")
    ax3.set_title("human alignment")

    plt.tight_layout()
    plt.savefig(save_dir / "xai_faithfulness_alignment.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved xai_faithfulness_alignment.png")

    # stability plot
    stab_methods = []
    for m in methods:
        if any("stability_" in k for k in xai[m]):
            stab_methods.append(m)

    if stab_methods:
        fig, ax = plt.subplots(figsize=(10, 5))
        # get perturbation names from first method
        perturb_keys = [k for k in xai[stab_methods[0]] if k.startswith("stability_") and k.endswith("_mean")]
        perturb_names = [k.replace("stability_", "").replace("_mean", "") for k in perturb_keys]

        x = np.arange(len(perturb_names))
        width = 0.8 / len(stab_methods)

        for i, m in enumerate(stab_methods):
            vals = [xai[m].get(f"stability_{p}_mean", 0) for p in perturb_names]
            ax.bar(x + i * width, vals, width, label=m)

        ax.set_xticks(x + width * (len(stab_methods) - 1) / 2)
        ax.set_xticklabels(perturb_names, rotation=30, ha="right")
        ax.set_ylabel("cosine similarity")
        ax.set_title("stability under perturbations")
        ax.legend()
        ax.set_ylim(0, 1.1)

        plt.tight_layout()
        plt.savefig(save_dir / "xai_stability.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"saved xai_stability.png")


def main():
    save_dir = RUNS_DIR / "figures"
    save_dir.mkdir(exist_ok=True)

    # training curves for each mode
    for mode in ["multitask", "cls_only", "sal_only"]:
        hist_path = RUNS_DIR / f"history_{mode}.json"
        if hist_path.exists():
            plot_training_curves_from_history(hist_path, save_dir)
            print(f"  plotted {mode} curves")

    plot_baseline_comparison(save_dir)
    plot_xai_summary(save_dir)

    print(f"\nall figures saved to {save_dir}")


if __name__ == "__main__":
    main()
