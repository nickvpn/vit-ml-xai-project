import json
import numpy as np
from src.config import RUNS_DIR


def _load(name):
    p = RUNS_DIR / name
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _agg(rows, key, val):
    g = {}
    for r in rows:
        g.setdefault(r[key], []).append(r[val])
    return [(k, float(np.mean(v)), float(np.std(v))) for k, v in sorted(g.items())]


def main():
    print("=" * 64)
    print("Intro to ML Project 2 -- consolidated results")
    print("=" * 64)

    bs = _load("baselines.json")
    if bs is not None:
        print("\n[baselines]")
        print(f"  sigma_rbf = {bs['sigma_rbf']:.4f}")
        for k, v in bs.items():
            if k.startswith("sigma"):
                continue
            print(f"  {k:18s} mse={v['mse']:.4f} r2={v['r2']:.4f}")

    rf = _load("random_feature_ridge.json")
    if rf is not None:
        print("\n[random feature ridge -- test MSE by P]")
        agg = _agg(rf["rows"], "p", "test_mse")
        for p, m, s in agg:
            mark = "  <-- N" if p == rf["n_train"] else ""
            print(f"  P={p:5d}: {m:.4f} (±{s:.4f}){mark}")

    tfa = _load("three_factor_ablation.json")
    if tfa is not None:
        print("\n[three-factor ablation summaries (test MSE at P=N)]")
        n = tfa["n_train"]
        for sec_key, sec_name in [("factor_a_regularization", "(a) ridge"),
                                    ("factor_b_projection", "(b) projection"),
                                    ("factor_c_noiseless", "(c) noiseless")]:
            print(f"  {sec_name}:")
            by_abl = {}
            for r in tfa[sec_key]:
                if r["p"] == n:
                    by_abl.setdefault(r["ablation"], []).append(r["test_mse"])
            for abl, vs in sorted(by_abl.items()):
                m, s = float(np.mean(vs)), float(np.std(vs))
                print(f"    {abl:30s} {m:.4f} ± {s:.4f}")

    kp = _load("kernel_pca_spectral.json")
    if kp is not None:
        print("\n[kernel pca / spectrum -- smallest singular value]")
        for key, rec in sorted(kp["spectra"].items(), key=lambda kv: int(kv[0])):
            print(f"  P={rec['p']:5d}: smallest sv={rec['smallest']:.5f} | "
                  f"ratio={rec['ratio_smallest_to_largest']:.2e}")


if __name__ == "__main__":
    main()
