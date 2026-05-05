import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import (
    RUNS_DIR, GRID_SIZE, NUM_LABELS, ANALYSIS_BATCH_SIZE, SEED, DEVICE
)
from src.seed import set_seed
from src.utils.load_models import load_variant
from src.representation.activations import build_val_loader
from src.utils.metrics import compute_cc, compute_sim


def kernel_shap_patch_grid(model, x, target_class, num_samples=300, grid=GRID_SIZE,
                            device=DEVICE):
    # kernel-shap-style attribution at the 14x14 patch level. uses uniform
    # subset-size sampling with shapley-kernel regression weights, plus an
    # equality constraint sum(phi) = f(full) - f(empty).
    n_patches = grid * grid
    img_size = x.shape[-1]
    M = n_patches

    rng = np.random.RandomState(SEED)
    # uniform size sampling in [1, M-1]
    sizes = rng.randint(low=1, high=M, size=num_samples)
    masks = np.zeros((num_samples, n_patches), dtype=np.float32)
    for i, s in enumerate(sizes):
        idx = rng.choice(n_patches, size=int(s), replace=False)
        masks[i, idx] = 1.0

    # shapley regression weights w(s) = (M-1) / (C(M,s) * s * (M-s)),
    # then normalized so the largest weight is 1 (numerical stability).
    from math import lgamma, log
    log_w = np.zeros(num_samples, dtype=np.float64)
    for i, s in enumerate(sizes):
        log_choose = lgamma(M + 1) - lgamma(s + 1) - lgamma(M - s + 1)
        log_w[i] = log(M - 1) - log_choose - log(s) - log(M - s)
    w = np.exp(log_w - log_w.max())

    outputs = np.zeros(num_samples, dtype=np.float32)
    bs = 32
    with torch.no_grad():
        for i in range(0, num_samples, bs):
            mb = torch.from_numpy(masks[i:i+bs]).to(device)
            m2d = mb.view(-1, 1, grid, grid)
            m2d = torch.nn.functional.interpolate(m2d, size=(img_size, img_size),
                                                    mode="nearest")
            xb = x.to(device).expand(mb.shape[0], -1, -1, -1) * m2d
            logits, _ = model(xb)
            outputs[i:i+bs] = torch.sigmoid(logits)[:, target_class].cpu().numpy()

    # boundary outputs anchor the constraint sum(phi) = f_full - f_empty
    with torch.no_grad():
        x_dev = x.to(device)
        f_full = float(torch.sigmoid(model(x_dev)[0])[:, target_class].cpu().numpy()[0])
        f_empty = float(torch.sigmoid(model(torch.zeros_like(x_dev))[0])[:, target_class].cpu().numpy()[0])

    # weighted least squares with equality constraint
    y = outputs - f_empty
    target_sum = f_full - f_empty
    Wsq = np.sqrt(w)
    A = masks * Wsq[:, None]
    yw = y * Wsq
    # phi_unc = argmin ||A phi - yw||^2 with small ridge for stability
    AtA = A.T @ A + 1e-3 * np.eye(M)
    Aty = A.T @ yw
    AtA_inv = np.linalg.inv(AtA)
    phi_unc = AtA_inv @ Aty
    # apply efficiency constraint: phi = phi_unc - mu * (AtA_inv @ 1) where mu chosen
    # so sum(phi) = target_sum
    ones = np.ones(M)
    denom = ones @ AtA_inv @ ones
    mu = (ones @ phi_unc - target_sum) / max(denom, 1e-9)
    phi = phi_unc - mu * (AtA_inv @ ones)
    return phi.reshape(grid, grid)


def deletion_insertion_auc(model, x, target_class, attribution, grid=GRID_SIZE,
                            device=DEVICE):
    # delete patches in decreasing |attribution| order, track sigmoid output for target_class.
    flat = attribution.flatten()
    order = np.argsort(-np.abs(flat))
    n = grid * grid
    img_size = x.shape[-1]

    masks_del = np.ones((n + 1, n), dtype=np.float32)
    masks_ins = np.zeros((n + 1, n), dtype=np.float32)
    for step, idx in enumerate(order, start=1):
        masks_del[step] = masks_del[step - 1]
        masks_del[step, idx] = 0.0
        masks_ins[step] = masks_ins[step - 1]
        masks_ins[step, idx] = 1.0

    def run_curve(masks):
        scores = np.zeros(masks.shape[0], dtype=np.float32)
        bs = 32
        with torch.no_grad():
            for i in range(0, masks.shape[0], bs):
                mb = torch.from_numpy(masks[i:i+bs]).to(device)
                m2d = mb.view(-1, 1, grid, grid)
                m2d = torch.nn.functional.interpolate(m2d, size=(img_size, img_size),
                                                       mode="nearest")
                xb = x.to(device).expand(mb.shape[0], -1, -1, -1) * m2d
                logits, _ = model(xb)
                p = torch.sigmoid(logits)[:, target_class].cpu().numpy()
                scores[i:i+bs] = p
        # auc as mean (trapz on uniform x is mean for monotone steps)
        return float(np.trapz(scores) / n)

    return run_curve(masks_del), run_curve(masks_ins)


def main(n_samples=100, shap_samples=200):
    set_seed(SEED)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # use multitask variant for the class-method baseline
    print("=== shap (kernel-shap, patch-grid) on multitask variant ===", flush=True)
    model = load_variant("multitask", device=DEVICE)
    print("building loader...", flush=True)
    loader = build_val_loader(n_samples=n_samples, batch_size=1)
    print(f"loader built, n_samples={n_samples}", flush=True)

    import time
    results = []
    t0 = time.time()
    for i, batch in enumerate(loader):
        x = batch["image"]
        sal_gt = batch["sal_full"][0, 0].numpy()  # (224, 224)
        with torch.no_grad():
            logits, _ = model(x.to(DEVICE))
        target = int(torch.argmax(logits, dim=-1).item())
        attribution = kernel_shap_patch_grid(model, x, target, num_samples=shap_samples)

        # downsample sal_gt to 14x14 for alignment
        from torch.nn.functional import adaptive_avg_pool2d
        sal_t = torch.from_numpy(sal_gt).float().unsqueeze(0).unsqueeze(0)
        sal_low = adaptive_avg_pool2d(sal_t, GRID_SIZE).numpy()[0, 0]

        # CC is sign-aware; pass raw attribution
        cc = compute_cc(attribution, sal_low)
        # SIM expects non-negative inputs that can be normalized to distributions
        attr_pos = attribution - attribution.min()
        sim = compute_sim(attr_pos, sal_low)
        del_auc, ins_auc = deletion_insertion_auc(model, x, target, attribution)

        results.append({
            "img_id": int(batch["img_id"][0]),
            "target_class": target,
            "del_auc": del_auc,
            "ins_auc": ins_auc,
            "align_cc": cc,
            "align_sim": sim,
        })
        if i % 10 == 0 or i == 0:
            elapsed = time.time() - t0
            print(f"sample {i+1}/{n_samples} | del={del_auc:.3f} ins={ins_auc:.3f} "
                  f"cc={cc:.3f} sim={sim:.3f} | t={elapsed:.1f}s", flush=True)

    summary = {
        "mean_del_auc": float(np.mean([r["del_auc"] for r in results])),
        "mean_ins_auc": float(np.mean([r["ins_auc"] for r in results])),
        "mean_align_cc": float(np.mean([r["align_cc"] for r in results])),
        "mean_align_sim": float(np.mean([r["align_sim"] for r in results])),
        "std_align_cc": float(np.std([r["align_cc"] for r in results])),
        "n_samples": len(results),
    }
    print("\nsummary:", summary)

    out = RUNS_DIR / "shap_results.json"
    with open(out, "w") as f:
        json.dump({"summary": summary, "per_sample": results}, f, indent=2)
    print(f"results saved to {out}")


if __name__ == "__main__":
    main()
