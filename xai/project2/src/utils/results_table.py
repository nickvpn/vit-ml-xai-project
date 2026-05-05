import json
from pathlib import Path
from src.config import RUNS_DIR


def _load(name):
    p = RUNS_DIR / name
    if not p.exists():
        print(f"  (missing {name})")
        return None
    with open(p) as f:
        return json.load(f)


def main():
    print("=" * 64)
    print("XAI Project 2 -- consolidated results")
    print("=" * 64)

    probes = _load("probes.json")
    if probes is not None:
        print("\n[probes] final-layer linear probe scores")
        print(f"{'variant':12s} {'class mAP':>10s} {'sal R2':>8s} {'loc acc':>8s} {'rand ctrl':>10s}")
        for mode, r in probes.items():
            print(f"{mode:12s} "
                  f"{r['class_map'][-1]:10.3f} "
                  f"{r['saliency_r2'][-1]:8.3f} "
                  f"{r['location_acc'][-1]:8.3f} "
                  f"{r['random_control_acc'][-1]:10.3f}")

    cka = _load("cka.json")
    if cka is not None:
        print("\n[cka] mid- and final-layer similarities (debiased)")
        print(f"{'pair':30s} {'layer':>6s} {'cka_lin':>8s} {'cka_rbf':>8s} "
              f"{'proc':>8s} {'svcca':>8s}")
        for key, r in cka.items():
            for li in [5, len(r["layers"]) - 1]:
                print(f"{key:30s} {r['layers'][li]:6d} "
                      f"{r['cka_linear'][li]:8.3f} {r['cka_rbf'][li]:8.3f} "
                      f"{r['procrustes'][li]:8.3f} {r['svcca'][li]:8.3f}")

    cluster = _load("clustering.json")
    if cluster is not None:
        print("\n[clustering] final-layer patch-token cluster quality")
        print(f"{'variant':12s} {'cat purity':>11s} {'cat nmi':>9s} "
              f"{'sal purity':>11s} {'sal nmi':>9s} {'stab ari':>9s}")
        for mode, r in cluster.items():
            print(f"{mode:12s} "
                  f"{r['category_purity']:11.3f} "
                  f"{r['category_nmi']:9.3f} "
                  f"{r['saliency_quartile_purity']:11.3f} "
                  f"{r['saliency_quartile_nmi']:9.3f} "
                  f"{r['seed_stability_ari']:9.3f}")

    sae = _load("sae_results.json")
    if sae is not None:
        print("\n[sae] reconstruction baseline")
        b = sae["sae_recon_baseline"]
        print(f"  cls diff (sae vs orig): {b['cls_diff']:.5f}")
        print(f"  sal diff (sae vs orig): {b['sal_diff']:.5f}")
        print("\n[sae] top features by ablation effect on cls and sal")
        rows = sae["ablation"]
        rows = sorted(rows, key=lambda r: -(r["delta_cls_mean_abs"] + r["delta_sal_mean_abs"]))
        print(f"  {'feat':>6s} {'frac':>7s} {'dCls':>9s} {'dSal':>9s}")
        for r in rows[:10]:
            print(f"  {r['feature']:6d} {r['frac_active']:7.4f} "
                  f"{r['delta_cls_mean_abs']:9.5f} {r['delta_sal_mean_abs']:9.5f}")

    shap = _load("shap_results.json")
    if shap is not None:
        s = shap["summary"]
        print("\n[shap, additional baseline] (multitask, kernel-shap, patch grid)")
        print(f"  del AUC = {s['mean_del_auc']:.3f} | ins AUC = {s['mean_ins_auc']:.3f}")
        print(f"  align CC = {s['mean_align_cc']:.3f} ({s['std_align_cc']:.3f} std) "
              f"| align SIM = {s['mean_align_sim']:.3f}")
        print(f"  n_samples = {s['n_samples']}")

    gc = _load("grad_cam_results.json")
    if gc is not None:
        s = gc["summary"]
        print("\n[grad-cam, class-method baseline (Lecture 9)] (multitask, block 10)")
        print(f"  del AUC = {s['mean_del_auc']:.3f} | ins AUC = {s['mean_ins_auc']:.3f}")
        print(f"  align CC = {s['mean_align_cc']:.3f} ({s['std_align_cc']:.3f} std) "
              f"| align SIM = {s['mean_align_sim']:.3f}")
        print(f"  n_samples = {s['n_samples']}")


if __name__ == "__main__":
    main()
