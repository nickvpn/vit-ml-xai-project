import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import RUNS_DIR


def print_all_results():
    print("=" * 70)
    print("CONSOLIDATED RESULTS")
    print("=" * 70)

    # baseline results
    baseline_path = RUNS_DIR / "baseline_results.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            baselines = json.load(f)

        print("\n--- CLASSIFICATION BASELINES ---")
        print(f"{'method':<25} {'mAP':>10}")
        print("-" * 35)
        for method, res in baselines["classification"].items():
            print(f"{method:<25} {res['mAP']:>10.4f}")

        print("\n--- SALIENCY BASELINES ---")
        print(f"{'method':<25} {'CC':>10} {'SIM':>10}")
        print("-" * 45)
        for method, res in baselines["saliency"].items():
            print(f"{method:<25} {res['CC']:>10.4f} {res['SIM']:>10.4f}")

    # model evaluation results
    print("\n--- TRAINED MODEL EVALUATION ---")
    for mode in ["multitask", "cls_only", "sal_only"]:
        eval_path = RUNS_DIR / f"eval_{mode}.json"
        if eval_path.exists():
            with open(eval_path) as f:
                results = json.load(f)
            print(f"\n  {mode}:")
            for k, v in results.items():
                print(f"    {k}: {v:.4f}")

    # xai results
    xai_path = RUNS_DIR / "xai_results.json"
    if xai_path.exists():
        with open(xai_path) as f:
            xai = json.load(f)

        print("\n--- XAI ANALYSIS ---")
        print(f"\n{'method':<25} {'del AUC':>10} {'ins AUC':>10} {'CC':>10} {'SIM':>10}")
        print("-" * 65)
        for method, res in xai.items():
            print(f"{method:<25} {res['deletion_auc_mean']:>10.4f} "
                  f"{res['insertion_auc_mean']:>10.4f} "
                  f"{res['alignment_CC_mean']:>10.4f} "
                  f"{res['alignment_SIM_mean']:>10.4f}")

        # stability
        stab_methods = [m for m in xai if any("stability_" in k for k in xai[m])]
        if stab_methods:
            print(f"\n--- STABILITY ---")
            for m in stab_methods:
                print(f"\n  {m}:")
                stab_keys = [k for k in xai[m] if k.startswith("stability_") and k.endswith("_mean")]
                for k in stab_keys:
                    pname = k.replace("stability_", "").replace("_mean", "")
                    print(f"    {pname}: {xai[m][k]:.4f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    print_all_results()
