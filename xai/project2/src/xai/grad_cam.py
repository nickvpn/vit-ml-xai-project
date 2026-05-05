import json
import numpy as np
import torch
import torch.nn.functional as F

from src.config import (
    RUNS_DIR, GRID_SIZE, NUM_LABELS, SEED, DEVICE
)
from src.seed import set_seed
from src.utils.load_models import load_variant
from src.representation.activations import build_val_loader
from src.utils.metrics import compute_cc, compute_sim


def grad_cam_patch_grid(model, x, target_class, layer_idx=None, grid=GRID_SIZE,
                         device=DEVICE):
    # selvaraju et al. 2017, adapted for vit. we capture the residual-stream
    # activations and gradients at the target block, average gradients over
    # the (b, tokens) batch+token axes per channel to get channel weights,
    # multiply each channel's activation map by its weight, sum over channels,
    # ReLU, and reshape the 196 patch tokens into a 14x14 map.
    #
    # for vit we cannot use the last block: patch tokens at the last block
    # do not feed the classification head (only the cls token does), so the
    # patch-token gradient w.r.t. the class logit is zero. we default to the
    # second-to-last block, where patch tokens still influence the cls token
    # through one more self-attention layer.
    model.eval()
    if layer_idx is None:
        layer_idx = len(model.vit.blocks) - 2

    feats = {}
    grads = {}

    def fhook(_m, _i, out):
        feats["v"] = out
        out.retain_grad()

    def bhook(_m, _gin, gout):
        grads["v"] = gout[0]

    block = model.vit.blocks[layer_idx]
    h_f = block.register_forward_hook(fhook)
    h_b = block.register_full_backward_hook(bhook)

    try:
        x = x.to(device)
        x.requires_grad_(False)
        logits, _ = model(x)
        logit = logits[0, target_class]
        model.zero_grad()
        logit.backward()

        a = feats["v"][0]  # (tokens, d) for batch=1
        g = grads["v"][0]  # (tokens, d)
        # drop the cls token; keep patch tokens
        a_p = a[1:]  # (196, d)
        g_p = g[1:]  # (196, d)
        # global-average-pool gradients over the spatial (token) axis to get
        # per-channel weights; this matches grad-cam's (h, w) -> per-channel pooling
        w = g_p.mean(dim=0)  # (d,)
        cam = (a_p * w[None, :]).sum(dim=-1)  # (196,)
        cam = F.relu(cam)
        cam = cam.detach().cpu().numpy().reshape(grid, grid)
    finally:
        h_f.remove()
        h_b.remove()

    return cam


def deletion_insertion_auc(model, x, target_class, attribution, grid=GRID_SIZE,
                            device=DEVICE):
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
                m2d = F.interpolate(m2d, size=(img_size, img_size), mode="nearest")
                xb = x.to(device).expand(mb.shape[0], -1, -1, -1) * m2d
                logits, _ = model(xb)
                p = torch.sigmoid(logits)[:, target_class].cpu().numpy()
                scores[i:i+bs] = p
        return float(np.trapz(scores) / n)

    return run_curve(masks_del), run_curve(masks_ins)


def main(n_samples=100):
    import time
    set_seed(SEED)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== grad-cam (lecture 9 class method) on multitask variant ===", flush=True)
    model = load_variant("multitask", device=DEVICE)
    print("building loader...", flush=True)
    loader = build_val_loader(n_samples=n_samples, batch_size=1)
    print(f"loader built, n_samples={n_samples}", flush=True)

    results = []
    t0 = time.time()
    for i, batch in enumerate(loader):
        x = batch["image"]
        sal_gt = batch["sal_full"][0, 0].numpy()
        with torch.no_grad():
            logits, _ = model(x.to(DEVICE))
        target = int(torch.argmax(logits, dim=-1).item())

        attribution = grad_cam_patch_grid(model, x, target, layer_idx=10)

        sal_t = torch.from_numpy(sal_gt).float().unsqueeze(0).unsqueeze(0)
        sal_low = F.adaptive_avg_pool2d(sal_t, GRID_SIZE).numpy()[0, 0]

        cc = compute_cc(attribution, sal_low)
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
            print(f"sample {i+1}/{n_samples} | del={del_auc:.3f} ins={ins_auc:.3f} "
                  f"cc={cc:.3f} sim={sim:.3f} | t={time.time()-t0:.1f}s", flush=True)

    summary = {
        "mean_del_auc": float(np.mean([r["del_auc"] for r in results])),
        "mean_ins_auc": float(np.mean([r["ins_auc"] for r in results])),
        "mean_align_cc": float(np.mean([r["align_cc"] for r in results])),
        "mean_align_sim": float(np.mean([r["align_sim"] for r in results])),
        "std_align_cc": float(np.std([r["align_cc"] for r in results])),
        "std_del_auc": float(np.std([r["del_auc"] for r in results])),
        "std_ins_auc": float(np.std([r["ins_auc"] for r in results])),
        "n_samples": len(results),
    }
    print("\nsummary:", summary)

    out = RUNS_DIR / "grad_cam_results.json"
    with open(out, "w") as f:
        json.dump({"summary": summary, "per_sample": results}, f, indent=2)
    print(f"results saved to {out}")


if __name__ == "__main__":
    main()
