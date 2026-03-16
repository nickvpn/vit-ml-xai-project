import torch
import numpy as np
from src.config import GRID_SIZE


def attention_rollout(model, image, device):
    # compute attention rollout across all transformer layers
    # returns a (14, 14) attribution map
    model.eval()
    image = image.to(device)

    attn_weights = model.get_attention_weights(image)

    # rollout: multiply attention matrices across layers
    # average across heads first
    result = None
    for attn in attn_weights:
        # attn: (b, heads, tokens, tokens)
        attn_avg = attn.mean(dim=1)  # (b, tokens, tokens)

        # add identity (residual connection)
        eye = torch.eye(attn_avg.size(-1), device=attn_avg.device)
        attn_avg = attn_avg + eye
        attn_avg = attn_avg / attn_avg.sum(dim=-1, keepdim=True)

        if result is None:
            result = attn_avg
        else:
            result = result @ attn_avg

    # take cls token attention to patch tokens
    # result: (b, tokens, tokens), cls token is index 0
    cls_attention = result[0, 0, 1:]  # skip cls token itself

    # reshape to grid
    rollout_map = cls_attention.reshape(GRID_SIZE, GRID_SIZE)

    return rollout_map.detach().cpu().numpy()
