import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    RUNS_DIR, NUM_BLOCKS, EMBED_DIM, ACTIVATION_SAMPLES,
    SAE_LAYER, SAE_DICT_SIZE, SAE_TOPK, SAE_EPOCHS, SAE_LR, SAE_BATCH,
    SEED, DEVICE
)
from src.seed import set_seed
from src.utils.load_models import load_variant
from src.representation.activations import build_val_loader, collect_activations


class TopKSAE(nn.Module):
    def __init__(self, d_in, d_dict, k):
        super().__init__()
        self.k = k
        self.encoder = nn.Linear(d_in, d_dict, bias=True)
        self.decoder = nn.Linear(d_dict, d_in, bias=False)
        # tie at init: decoder = encoder.T (then trained independently)
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.T.contiguous())
        self.b_pre = nn.Parameter(torch.zeros(d_in))

    def encode(self, x):
        # subtract pre-bias, encode, top-k mask
        z = self.encoder(x - self.b_pre)
        z = F.relu(z)
        topv, topi = torch.topk(z, k=self.k, dim=-1)
        mask = torch.zeros_like(z)
        mask.scatter_(-1, topi, topv)
        return mask

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decoder(z) + self.b_pre
        return x_hat, z


def collect_patch_activations(mode, layer, n_samples):
    model = load_variant(mode, device=DEVICE)
    loader = build_val_loader(n_samples=n_samples)
    feats, meta = collect_activations(model, loader, device=DEVICE,
                                      layers=[layer], token="patch_all")
    f = feats[layer]  # (n, 196, d)
    return f, meta, model


def train_sae(activations, d_dict=SAE_DICT_SIZE, k=SAE_TOPK,
              epochs=SAE_EPOCHS, lr=SAE_LR, batch=SAE_BATCH, device=DEVICE):
    # activations: (N_total_tokens, d_in)
    N, d = activations.shape
    sae = TopKSAE(d_in=d, d_dict=d_dict, k=k).to(device)
    opt = torch.optim.AdamW(sae.parameters(), lr=lr, weight_decay=0.0)

    X = torch.from_numpy(activations).float()
    ds = TensorDataset(X)
    loader = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0)

    history = {"epoch": [], "recon_loss": [], "frac_active": []}
    for ep in range(epochs):
        sae.train()
        running = 0.0
        running_active = 0.0
        nb = 0
        for (xb,) in loader:
            xb = xb.to(device)
            x_hat, z = sae(xb)
            loss = F.mse_loss(x_hat, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            running_active += (z > 0).float().mean().item()
            nb += 1
        avg = running / max(1, nb)
        avg_active = running_active / max(1, nb)
        history["epoch"].append(ep)
        history["recon_loss"].append(avg)
        history["frac_active"].append(avg_active)
        print(f"epoch {ep+1}/{epochs} | recon_mse={avg:.5f} | frac_active={avg_active:.4f}")

    return sae, history


@torch.no_grad()
def compute_feature_usage(sae, activations, batch=SAE_BATCH, device=DEVICE):
    # for each feature: fraction of tokens where it is active and mean activation
    sae.eval()
    X = torch.from_numpy(activations).float()
    n_tokens = X.shape[0]
    d_dict = sae.encoder.out_features
    active_count = torch.zeros(d_dict, device=device)
    sum_act = torch.zeros(d_dict, device=device)
    for i in range(0, n_tokens, batch):
        xb = X[i:i+batch].to(device)
        z = sae.encode(xb)
        active_count += (z > 0).float().sum(dim=0)
        sum_act += z.sum(dim=0)
    frac_active = (active_count / n_tokens).cpu().numpy()
    mean_act = (sum_act / (active_count + 1e-8)).cpu().numpy()
    return frac_active, mean_act


@torch.no_grad()
def head_outputs(model, x, device=DEVICE):
    # forward pass, return (logits, sal_grid)
    x = x.to(device)
    logits, sal = model(x)
    return logits.cpu(), sal.cpu()


def feature_ablation_effect(sae, model, activations, loader, layer, top_features,
                             baseline_metrics, device=DEVICE):
    # for each feature in top_features, replace its contribution with the SAE-reconstructed
    # baseline (no ablation) and ablate just that single feature, measure delta cls and sal
    # this is expensive, so we work on a small loader of full images
    # we hook block `layer` to substitute residual stream with SAE-reconstructed residual
    pass


def patch_residual_replace(model, loader, sae, layer, drop_feature=None, device=DEVICE):
    # forward each batch with a hook that replaces the layer-`layer` residual stream
    # patch tokens with the SAE reconstruction. if drop_feature is not None, that feature
    # is zeroed in the latent before decoding. returns (cls_logits, sal_grid) for all batches.
    sae.eval()
    cls_outs = []
    sal_outs = []

    def hook(module, inp, out):
        # out: (b, tokens, d). tokens[0] is CLS, the rest are 196 patches.
        b, t, d = out.shape
        x = out.clone()
        patch = x[:, 1:, :].reshape(-1, d)
        # encode + optionally drop a feature + decode
        z = sae.encode(patch)
        if drop_feature is not None:
            z[:, drop_feature] = 0.0
        recon = sae.decoder(z) + sae.b_pre
        x[:, 1:, :] = recon.reshape(b, t - 1, d)
        return x

    h = model.vit.blocks[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            for batch in loader:
                xb = batch["image"].to(device)
                logits, sal = model(xb)
                cls_outs.append(torch.sigmoid(logits).cpu())
                sal_outs.append(sal.cpu())
    finally:
        h.remove()

    return torch.cat(cls_outs, dim=0), torch.cat(sal_outs, dim=0)


def main(n_samples=ACTIVATION_SAMPLES, n_features_to_ablate=20,
          ablation_n_samples=200):
    set_seed(SEED)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== sae feature decomposition (multitask, layer {SAE_LAYER}) ===")

    # 1. collect activations from multitask variant
    feats, meta, model = collect_patch_activations("multitask", SAE_LAYER, n_samples)
    n, p, d = feats.shape
    print(f"activations: ({n}, {p}, {d})")
    flat = feats.reshape(-1, d)

    # 2. train sae on flattened patch tokens
    sae, history = train_sae(flat)

    # save sae weights
    sae_path = RUNS_DIR / f"sae_multitask_layer{SAE_LAYER}.pt"
    torch.save({"state_dict": sae.state_dict(),
                "d_in": d, "d_dict": SAE_DICT_SIZE, "k": SAE_TOPK,
                "history": history,
                "layer": SAE_LAYER}, sae_path)
    print(f"sae saved to {sae_path}")

    # 3. feature usage stats
    frac_active, mean_act = compute_feature_usage(sae, flat)
    feature_stats = {
        "frac_active": frac_active.tolist(),
        "mean_act": mean_act.tolist(),
    }

    # 4. ablation experiment: pick top-K features by usage and measure causal effect
    top_idx = np.argsort(-frac_active)[:n_features_to_ablate].tolist()

    # use a small held-out batch for ablation comparison
    ablation_loader = build_val_loader(n_samples=ablation_n_samples)

    # baseline: SAE-reconstructed residual stream with no feature dropped (anchored baseline)
    base_cls, base_sal = patch_residual_replace(model, ablation_loader, sae, SAE_LAYER,
                                                  drop_feature=None)

    # original: no SAE substitution at all (sanity reference)
    with torch.no_grad():
        orig_cls = []
        orig_sal = []
        for batch in ablation_loader:
            xb = batch["image"].to(DEVICE)
            logits, sal = model(xb)
            orig_cls.append(torch.sigmoid(logits).cpu())
            orig_sal.append(sal.cpu())
        orig_cls = torch.cat(orig_cls, dim=0)
        orig_sal = torch.cat(orig_sal, dim=0)

    sae_recon_cls_diff = float((base_cls - orig_cls).abs().mean().item())
    sae_recon_sal_diff = float((base_sal - orig_sal).abs().mean().item())
    print(f"sae reconstruction baseline | cls diff={sae_recon_cls_diff:.5f} sal diff={sae_recon_sal_diff:.5f}")

    # ablation effects relative to the SAE-reconstructed baseline
    ablation_results = []
    for fi in top_idx:
        a_cls, a_sal = patch_residual_replace(model, ablation_loader, sae, SAE_LAYER,
                                                drop_feature=fi)
        d_cls = float((a_cls - base_cls).abs().mean().item())
        d_sal = float((a_sal - base_sal).abs().mean().item())
        ablation_results.append({
            "feature": int(fi),
            "frac_active": float(frac_active[fi]),
            "delta_cls_mean_abs": d_cls,
            "delta_sal_mean_abs": d_sal,
        })
        print(f"feature {fi:5d} | active={frac_active[fi]:.4f} | dCls={d_cls:.5f} | dSal={d_sal:.5f}")

    out = RUNS_DIR / "sae_results.json"
    with open(out, "w") as f:
        json.dump({
            "history": history,
            "feature_stats": feature_stats,
            "sae_recon_baseline": {
                "cls_diff": sae_recon_cls_diff,
                "sal_diff": sae_recon_sal_diff,
            },
            "ablation": ablation_results,
            "config": {
                "layer": SAE_LAYER,
                "d_dict": SAE_DICT_SIZE,
                "k": SAE_TOPK,
                "epochs": SAE_EPOCHS,
                "ablation_n_samples": ablation_n_samples,
                "n_features_to_ablate": n_features_to_ablate,
            }
        }, f, indent=2)
    print(f"\nresults saved to {out}")


if __name__ == "__main__":
    main()
